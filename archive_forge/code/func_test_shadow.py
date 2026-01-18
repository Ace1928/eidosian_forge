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
def test_shadow(self):
    ctx = self.Context()
    ctx2 = self.Context.shadow(ctx.underlying)
    assert ctx.underlying == ctx2.underlying
    s = ctx.socket(zmq.PUB)
    s.close()
    del ctx2
    assert not ctx.closed
    s = ctx.socket(zmq.PUB)
    ctx2 = self.Context.shadow(ctx)
    with ctx2.socket(zmq.PUB) as s2:
        pass
    assert s2.closed
    assert not s.closed
    s.close()
    ctx3 = self.Context(ctx)
    assert ctx3.underlying == ctx.underlying
    del ctx3
    assert not ctx.closed
    ctx.term()
    self.assertRaisesErrno(zmq.EFAULT, ctx2.socket, zmq.PUB)
    del ctx2