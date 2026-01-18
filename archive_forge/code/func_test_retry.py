import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_retry(self):
    result = 'RESULT'

    @loopingcall.RetryDecorator()
    def func(*args, **kwargs):
        return result
    self.assertEqual(result, func())

    def func2(*args, **kwargs):
        return result
    retry = loopingcall.RetryDecorator()
    self.assertEqual(result, retry(func2)())
    self.assertTrue(retry._retry_count == 0)