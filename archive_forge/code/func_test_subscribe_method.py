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
def test_subscribe_method(self):
    pub, sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
    sub.subscribe('prefix')
    sub.subscribe = 'c'
    p = zmq.Poller()
    p.register(sub, zmq.POLLIN)
    for i in range(100):
        pub.send(b'canary')
        events = p.poll(250)
        if events:
            break
    self.recv(sub)
    pub.send(b'prefixmessage')
    msg = self.recv(sub)
    assert msg == b'prefixmessage'
    sub.unsubscribe('prefix')
    pub.send(b'prefixmessage')
    events = p.poll(1000)
    assert events == []