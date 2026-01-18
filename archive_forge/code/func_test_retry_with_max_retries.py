import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_retry_with_max_retries(self):
    responses = [AnException(None), AnException(None), AnException(None)]

    def func(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    retry = loopingcall.RetryDecorator(2, 0, 0, (AnException,))
    self.assertRaises(AnException, retry(func))
    self.assertTrue(retry._retry_count == 2)