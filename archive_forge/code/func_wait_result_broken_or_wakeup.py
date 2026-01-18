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