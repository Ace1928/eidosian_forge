import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
@pytest.mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason='requires RCVTIMEO')
def test_recv_timeout(self):

    async def test():
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        b.rcvtimeo = 100
        f1 = b.recv()
        b.rcvtimeo = 1000
        f2 = b.recv_multipart()
        with pytest.raises(zmq.Again):
            await f1
        await a.send_multipart([b'hi', b'there'])
        recvd = await f2
        assert f2.done()
        assert recvd == [b'hi', b'there']
    self.loop.run_sync(test)