import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_socket_class(self):
    s = self.context.socket(zmq.PUSH)
    assert isinstance(s, future.Socket)
    s.close()