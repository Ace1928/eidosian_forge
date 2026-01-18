import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy
def test_send_buffer(self):
    a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
    for buffer_type in (memoryview, bytearray):
        rawbytes = str(buffer_type).encode('ascii')
        msg = buffer_type(rawbytes)
        a.send(msg)
        recvd = b.recv()
        assert recvd == rawbytes