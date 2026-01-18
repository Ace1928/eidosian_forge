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
def test_term_hang(self):
    rep, req = self.create_bound_pair(zmq.ROUTER, zmq.DEALER)
    req.setsockopt(zmq.LINGER, 0)
    req.send(b'hello', copy=False)
    req.close()
    rep.close()
    self.context.term()