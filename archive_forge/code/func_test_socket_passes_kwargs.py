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
def test_socket_passes_kwargs(self):
    test_kwarg_value = 'testing one two three'
    with KwargTestContext() as ctx:
        with ctx.socket(zmq.DEALER, test_kwarg=test_kwarg_value) as socket:
            assert socket.test_kwarg_value is test_kwarg_value