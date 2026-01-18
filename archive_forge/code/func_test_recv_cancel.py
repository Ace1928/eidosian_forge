import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_recv_cancel(self):

    async def test():
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        f1 = b.recv()
        f2 = b.recv_multipart()
        assert f1.cancel()
        assert f1.done()
        assert not f2.done()
        await a.send_multipart([b'hi', b'there'])
        recvd = await f2
        assert f1.cancelled()
        assert f2.done()
        assert recvd == [b'hi', b'there']
    self.loop.run_sync(test)