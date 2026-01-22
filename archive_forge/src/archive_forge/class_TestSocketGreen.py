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
class TestSocketGreen(GreenTest, TestSocket):
    test_bad_attr = GreenTest.skip_green
    test_close_after_destroy = GreenTest.skip_green
    _repr_cls = 'zmq.green.Socket'

    def test_timeout(self):
        a, b = self.create_bound_pair()
        g = gevent.spawn_later(0.5, lambda: a.send(b'hi'))
        timeout = gevent.Timeout(0.1)
        timeout.start()
        self.assertRaises(gevent.Timeout, b.recv)
        g.kill()

    @mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason='requires RCVTIMEO')
    def test_warn_set_timeo(self):
        s = self.context.socket(zmq.REQ)
        with warnings.catch_warnings(record=True) as w:
            s.rcvtimeo = 5
        s.close()
        assert len(w) == 1
        assert w[0].category == UserWarning

    @mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason='requires SNDTIMEO')
    def test_warn_get_timeo(self):
        s = self.context.socket(zmq.REQ)
        with warnings.catch_warnings(record=True) as w:
            s.sndtimeo
        s.close()
        assert len(w) == 1
        assert w[0].category == UserWarning