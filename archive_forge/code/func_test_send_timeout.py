import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
@pytest.mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason='requires SNDTIMEO')
def test_send_timeout(self):

    async def test():
        s = self.socket(zmq.PUSH)
        s.sndtimeo = 100
        with pytest.raises(zmq.Again):
            await s.send(b'not going anywhere')
    self.loop.run_sync(test)