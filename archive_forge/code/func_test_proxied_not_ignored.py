import threading
import time
from unittest import mock
from oslo_config import fixture as config
from oslo_serialization import jsonutils
from oslotest import base as test_base
import requests
import webob.dec
import webob.exc
from oslo_middleware import healthcheck
from oslo_middleware.healthcheck import __main__
def test_proxied_not_ignored(self):
    conf = {}
    self._do_test(conf, expected_code=webob.exc.HTTPOk.code, headers={'Forwarded-For': 'http://localhost'})