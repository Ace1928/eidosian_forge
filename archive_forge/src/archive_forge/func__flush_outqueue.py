import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def _flush_outqueue(self, fd, remove, process_index, on_state_change):
    try:
        proc = process_index[fd]
    except KeyError:
        return remove(fd)
    reader = proc.outq._reader
    try:
        setblocking(reader, 1)
    except OSError:
        return remove(fd)
    try:
        if reader.poll(0):
            task = reader.recv()
        else:
            task = None
            sleep(0.5)
    except (OSError, EOFError):
        return remove(fd)
    else:
        if task:
            on_state_change(task)
    finally:
        try:
            setblocking(reader, 0)
        except OSError:
            return remove(fd)