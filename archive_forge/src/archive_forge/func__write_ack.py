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
def _write_ack(fd, ack, callback=None):
    header, body, body_size = ack[2]
    try:
        try:
            proc = fileno_to_synq[fd]
        except KeyError:
            raise StopIteration()
        send = proc.send_syn_offset
        Hw = Bw = 0
        while Hw < 4:
            try:
                Hw += send(header, Hw)
            except Exception as exc:
                if getattr(exc, 'errno', None) not in UNAVAIL:
                    raise
                yield
        while Bw < body_size:
            try:
                Bw += send(body, Bw)
            except Exception as exc:
                if getattr(exc, 'errno', None) not in UNAVAIL:
                    raise
                yield
    finally:
        if callback:
            callback()
        active_writes.discard(fd)