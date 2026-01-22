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
class KwargTestSocket(zmq.Socket):
    test_kwarg_value = None

    def __init__(self, *args, **kwargs):
        self.test_kwarg_value = kwargs.pop('test_kwarg', None)
        super().__init__(*args, **kwargs)