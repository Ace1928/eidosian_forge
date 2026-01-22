import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase
from pytest import mark
import zmq
from zmq.utils import jsonapi
def assertRaisesErrno(self, errno, func, *args, **kwargs):
    if errno == zmq.EAGAIN:
        raise SkipTest("Skipping because we're green.")
    try:
        func(*args, **kwargs)
    except zmq.ZMQError:
        e = sys.exc_info()[1]
        self.assertEqual(e.errno, errno, f"wrong error raised, expected '{zmq.ZMQError(errno)}' got '{zmq.ZMQError(e.errno)}'")
    else:
        self.fail('Function did not raise any error')