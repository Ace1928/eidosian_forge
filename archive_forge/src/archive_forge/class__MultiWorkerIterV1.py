import pickle
import io
import sys
import signal
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
from . import sampler as _sampler
from ... import nd, context
from ...util import is_np_shape, is_np_array, set_np
from ... import numpy as _mx_np  # pylint: disable=reimported
class _MultiWorkerIterV1(object):
    """Internal multi-worker iterator for DataLoader."""

    def __init__(self, num_workers, dataset, batchify_fn, batch_sampler, pin_memory=False, pin_device_id=0, worker_fn=worker_loop_v1):
        assert num_workers > 0, '_MultiWorkerIter is not for {} workers'.format(num_workers)
        self._num_workers = num_workers
        self._dataset = dataset
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._key_queue = Queue()
        self._data_queue = SimpleQueue()
        self._data_buffer = {}
        self._data_buffer_lock = threading.Lock()
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._shutdown = False
        workers = []
        for _ in range(self._num_workers):
            worker = multiprocessing.Process(target=worker_fn, args=(self._dataset, self._key_queue, self._data_queue, self._batchify_fn))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        self._workers = workers
        self._fetcher = threading.Thread(target=fetcher_loop_v1, args=(self._data_queue, self._data_buffer, pin_memory, pin_device_id, self._data_buffer_lock))
        self._fetcher.daemon = True
        self._fetcher.start()
        for _ in range(2 * self._num_workers):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def __del__(self):
        self.shutdown()

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        self._key_queue.put((self._sent_idx, r))
        self._sent_idx += 1

    def __next__(self):
        assert not self._shutdown, 'call __next__ after shutdown is forbidden'
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            self.shutdown()
            raise StopIteration
        while True:
            if self._rcvd_idx in self._data_buffer:
                with self._data_buffer_lock:
                    batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                self._push_next()
                return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def shutdown(self):
        """Shutdown internal workers by pushing terminate signals."""
        if not self._shutdown:
            self._data_queue.put((None, None))
            self._fetcher.join()
            for _ in range(self._num_workers):
                self._key_queue.put((None, None))
            for w in self._workers:
                if w.is_alive():
                    w.terminate()
            self._shutdown = True