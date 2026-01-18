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
@mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason='requires SNDTIMEO')
def test_warn_get_timeo(self):
    s = self.context.socket(zmq.REQ)
    with warnings.catch_warnings(record=True) as w:
        s.sndtimeo
    s.close()
    assert len(w) == 1
    assert w[0].category == UserWarning