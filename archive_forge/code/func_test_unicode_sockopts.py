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
def test_unicode_sockopts(self):
    """test setting/getting sockopts with unicode strings"""
    topic = 'tést'
    p, s = self.create_bound_pair(zmq.PUB, zmq.SUB)
    assert s.send_unicode == s.send_unicode
    assert p.recv_unicode == p.recv_unicode
    self.assertRaises(TypeError, s.setsockopt, zmq.SUBSCRIBE, topic)
    self.assertRaises(TypeError, s.setsockopt, zmq.IDENTITY, topic)
    s.setsockopt_unicode(zmq.IDENTITY, topic, 'utf16')
    self.assertRaises(TypeError, s.setsockopt, zmq.AFFINITY, topic)
    s.setsockopt_unicode(zmq.SUBSCRIBE, topic)
    self.assertRaises(TypeError, s.getsockopt_unicode, zmq.AFFINITY)
    self.assertRaisesErrno(zmq.EINVAL, s.getsockopt_unicode, zmq.SUBSCRIBE)
    identb = s.getsockopt(zmq.IDENTITY)
    identu = identb.decode('utf16')
    identu2 = s.getsockopt_unicode(zmq.IDENTITY, 'utf16')
    assert identu == identu2
    time.sleep(0.1)
    p.send_unicode(topic, zmq.SNDMORE)
    p.send_unicode(topic * 2, encoding='latin-1')
    assert topic == s.recv_unicode()
    assert topic * 2 == s.recv_unicode(encoding='latin-1')