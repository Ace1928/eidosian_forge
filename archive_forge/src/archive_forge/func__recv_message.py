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
def _recv_message(self, add_reader, fd, callback, __read__=__read__, readcanbuf=readcanbuf, BytesIO=BytesIO, unpack_from=unpack_from, load=_pickle.load):
    Hr = Br = 0
    if readcanbuf:
        buf = bytearray(4)
        bufv = memoryview(buf)
    else:
        buf = bufv = BytesIO()
    while Hr < 4:
        try:
            n = __read__(fd, bufv[Hr:] if readcanbuf else bufv, 4 - Hr)
        except OSError as exc:
            if exc.errno not in UNAVAIL:
                raise
            yield
        else:
            if n == 0:
                raise OSError('End of file during message') if Hr else EOFError()
            Hr += n
    body_size, = unpack_from('>i', bufv)
    if readcanbuf:
        buf = bytearray(body_size)
        bufv = memoryview(buf)
    else:
        buf = bufv = BytesIO()
    while Br < body_size:
        try:
            n = __read__(fd, bufv[Br:] if readcanbuf else bufv, body_size - Br)
        except OSError as exc:
            if exc.errno not in UNAVAIL:
                raise
            yield
        else:
            if n == 0:
                raise OSError('End of file during message') if Br else EOFError()
            Br += n
    add_reader(fd, self.handle_event, fd)
    if readcanbuf:
        message = load(BytesIO(bufv))
    else:
        bufv.seek(0)
        message = load(bufv)
    if message:
        callback(message)