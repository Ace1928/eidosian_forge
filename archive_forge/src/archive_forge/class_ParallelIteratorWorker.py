import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@Deprecated
class ParallelIteratorWorker(object):
    """Worker actor for a ParallelIterator.

    Actors that are passed to iter.from_actors() must subclass this interface.
    """

    def __init__(self, item_generator: Any, repeat: bool):
        """Create an iterator worker.

        Subclasses must call this init function.

        Args:
            item_generator: A Python iterable or lambda function
                that produces a generator when called. We allow lambda
                functions since the generator itself might not be serializable,
                but a lambda that returns it can be.
            repeat: Whether to loop over the iterator forever.
        """

        def make_iterator():
            if callable(item_generator):
                return item_generator()
            else:
                return item_generator
        if repeat:

            def cycle():
                while True:
                    it = iter(make_iterator())
                    if it is item_generator:
                        raise ValueError('Cannot iterate over {0} multiple times.' + 'Please pass in the base iterable or' + 'lambda: {0} instead.'.format(item_generator))
                    for item in it:
                        yield item
            self.item_generator = cycle()
        else:
            self.item_generator = make_iterator()
        self.transforms = []
        self.local_it = None
        self.next_ith_buffer = None

    def par_iter_init(self, transforms):
        """Implements ParallelIterator worker init."""
        it = LocalIterator(lambda timeout: self.item_generator, SharedMetrics())
        for fn in transforms:
            it = fn(it)
            assert it is not None, fn
        self.local_it = iter(it)

    def par_iter_next(self):
        """Implements ParallelIterator worker item fetch."""
        assert self.local_it is not None, 'must call par_iter_init()'
        return next(self.local_it)

    def par_iter_next_batch(self, batch_ms: int):
        """Batches par_iter_next."""
        batch = []
        if batch_ms == 0:
            batch.append(self.par_iter_next())
            return batch
        t_end = time.time() + 0.001 * batch_ms
        while time.time() < t_end:
            try:
                batch.append(self.par_iter_next())
            except StopIteration:
                if len(batch) == 0:
                    raise StopIteration
                else:
                    pass
        return batch

    def par_iter_slice(self, step: int, start: int):
        """Iterates in increments of step starting from start."""
        assert self.local_it is not None, 'must call par_iter_init()'
        if self.next_ith_buffer is None:
            self.next_ith_buffer = collections.defaultdict(list)
        index_buffer = self.next_ith_buffer[start]
        if len(index_buffer) > 0:
            return index_buffer.pop(0)
        else:
            for j in range(step):
                try:
                    val = next(self.local_it)
                    self.next_ith_buffer[j].append(val)
                except StopIteration:
                    pass
            if not self.next_ith_buffer[start]:
                raise StopIteration
        return self.next_ith_buffer[start].pop(0)

    def par_iter_slice_batch(self, step: int, start: int, batch_ms: int):
        """Batches par_iter_slice."""
        batch = []
        if batch_ms == 0:
            batch.append(self.par_iter_slice(step, start))
            return batch
        t_end = time.time() + 0.001 * batch_ms
        while time.time() < t_end:
            try:
                batch.append(self.par_iter_slice(step, start))
            except StopIteration:
                if len(batch) == 0:
                    raise StopIteration
                else:
                    pass
        return batch