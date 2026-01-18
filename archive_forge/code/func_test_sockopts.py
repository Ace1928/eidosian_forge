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
def test_sockopts(self):
    """setting socket options with ctx attributes"""
    ctx = self.Context()
    ctx.linger = 5
    assert ctx.linger == 5
    s = ctx.socket(zmq.REQ)
    assert s.linger == 5
    assert s.getsockopt(zmq.LINGER) == 5
    s.close()
    ctx.subscribe = b''
    s = ctx.socket(zmq.REQ)
    s.close()
    ctx.term()