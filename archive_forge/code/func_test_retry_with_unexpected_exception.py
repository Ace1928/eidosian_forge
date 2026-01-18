import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_retry_with_unexpected_exception(self):

    def func(*args, **kwargs):
        raise UnknownException(None)
    retry = loopingcall.RetryDecorator()
    self.assertRaises(UnknownException, retry(func))
    self.assertTrue(retry._retry_count == 0)