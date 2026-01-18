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
def test_recv_multipart(self):
    a, b = self.create_bound_pair()
    msg = b'hi'
    for i in range(3):
        a.send(msg)
    time.sleep(0.1)
    for i in range(3):
        assert self.recv_multipart(b) == [msg]