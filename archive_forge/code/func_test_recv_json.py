import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_recv_json(self):

    async def test():
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        f = b.recv_json()
        assert not f.done()
        obj = dict(a=5)
        await a.send_json(obj)
        recvd = await f
        assert f.done()
        assert f.result() == obj
        assert recvd == obj
    self.loop.run_sync(test)