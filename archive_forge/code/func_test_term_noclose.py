import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
def test_term_noclose(self):
    """Context.term won't close sockets"""
    ctx = self.Context()
    s = ctx.socket(zmq.REQ)
    assert not s.closed
    t = Thread(target=ctx.term)
    t.start()
    t.join(timeout=0.1)
    assert t.is_alive(), 'Context should be waiting'
    s.close()
    t.join(timeout=0.1)
    assert not t.is_alive(), 'Context should have closed'