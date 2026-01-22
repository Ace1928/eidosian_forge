import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """

    def __init__(self, executor):
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock

        def weakref_cb(_, thread_wakeup=self.thread_wakeup, shutdown_lock=self.shutdown_lock):
            mp.util.debug('Executor collected: triggering callback for QueueManager wakeup')
            with shutdown_lock:
                thread_wakeup.wakeup()
        self.executor_reference = weakref.ref(executor, weakref_cb)
        self.processes = executor._processes
        self.call_queue = executor._call_queue
        self.result_queue = executor._result_queue
        self.work_ids_queue = executor._work_ids
        self.max_tasks_per_child = executor._max_tasks_per_child
        self.pending_work_items = executor._pending_work_items
        super().__init__()

    def run(self):
        while True:
            self.add_call_item_to_queue()
            result_item, is_broken, cause = self.wait_result_broken_or_wakeup()
            if is_broken:
                self.terminate_broken(cause)
                return
            if result_item is not None:
                self.process_result_item(result_item)
                process_exited = result_item.exit_pid is not None
                if process_exited:
                    p = self.processes.pop(result_item.exit_pid)
                    p.join()
                del result_item
                if (executor := self.executor_reference()):
                    if process_exited:
                        with self.shutdown_lock:
                            executor._adjust_process_count()
                    else:
                        executor._idle_worker_semaphore.release()
                    del executor
            if self.is_shutting_down():
                self.flag_executor_shutting_down()
                self.add_call_item_to_queue()
                if not self.pending_work_items:
                    self.join_executor_internals()
                    return

    def add_call_item_to_queue(self):
        while True:
            if self.call_queue.full():
                return
            try:
                work_id = self.work_ids_queue.get(block=False)
            except queue.Empty:
                return
            else:
                work_item = self.pending_work_items[work_id]
                if work_item.future.set_running_or_notify_cancel():
                    self.call_queue.put(_CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs), block=True)
                else:
                    del self.pending_work_items[work_id]
                    continue

    def wait_result_broken_or_wakeup(self):
        result_reader = self.result_queue._reader
        assert not self.thread_wakeup._closed
        wakeup_reader = self.thread_wakeup._reader
        readers = [result_reader, wakeup_reader]
        worker_sentinels = [p.sentinel for p in list(self.processes.values())]
        ready = mp.connection.wait(readers + worker_sentinels)
        cause = None
        is_broken = True
        result_item = None
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                is_broken = False
            except BaseException as e:
                cause = format_exception(type(e), e, e.__traceback__)
        elif wakeup_reader in ready:
            is_broken = False
        self.thread_wakeup.clear()
        return (result_item, is_broken, cause)

    def process_result_item(self, result_item):
        if isinstance(result_item, int):
            assert self.is_shutting_down()
            p = self.processes.pop(result_item)
            p.join()
            if not self.processes:
                self.join_executor_internals()
                return
        else:
            work_item = self.pending_work_items.pop(result_item.work_id, None)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)

    def is_shutting_down(self):
        executor = self.executor_reference()
        return _global_shutdown or executor is None or executor._shutdown_thread

    def terminate_broken(self, cause):
        executor = self.executor_reference()
        if executor is not None:
            executor._broken = 'A child process terminated abruptly, the process pool is not usable anymore'
            executor._shutdown_thread = True
            executor = None
        bpe = BrokenProcessPool('A process in the process pool was terminated abruptly while the future was running or pending.')
        if cause is not None:
            bpe.__cause__ = _RemoteTraceback(f"\n'''\n{''.join(cause)}'''")
        for work_id, work_item in self.pending_work_items.items():
            work_item.future.set_exception(bpe)
            del work_item
        self.pending_work_items.clear()
        for p in self.processes.values():
            p.terminate()
        self.call_queue._reader.close()
        if sys.platform == 'win32':
            self.call_queue._writer.close()
        self.join_executor_internals()

    def flag_executor_shutting_down(self):
        executor = self.executor_reference()
        if executor is not None:
            executor._shutdown_thread = True
            if executor._cancel_pending_futures:
                new_pending_work_items = {}
                for work_id, work_item in self.pending_work_items.items():
                    if not work_item.future.cancel():
                        new_pending_work_items[work_id] = work_item
                self.pending_work_items = new_pending_work_items
                while True:
                    try:
                        self.work_ids_queue.get_nowait()
                    except queue.Empty:
                        break
                executor._cancel_pending_futures = False

    def shutdown_workers(self):
        n_children_to_stop = self.get_n_children_alive()
        n_sentinels_sent = 0
        while n_sentinels_sent < n_children_to_stop and self.get_n_children_alive() > 0:
            for i in range(n_children_to_stop - n_sentinels_sent):
                try:
                    self.call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except queue.Full:
                    break

    def join_executor_internals(self):
        self.shutdown_workers()
        self.call_queue.close()
        self.call_queue.join_thread()
        with self.shutdown_lock:
            self.thread_wakeup.close()
        for p in self.processes.values():
            p.join()

    def get_n_children_alive(self):
        return sum((p.is_alive() for p in self.processes.values()))