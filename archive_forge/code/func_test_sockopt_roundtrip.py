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
def test_sockopt_roundtrip(self):
    """test set/getsockopt roundtrip."""
    p = self.context.socket(zmq.PUB)
    self.sockets.append(p)
    p.setsockopt(zmq.LINGER, 11)
    assert p.getsockopt(zmq.LINGER) == 11