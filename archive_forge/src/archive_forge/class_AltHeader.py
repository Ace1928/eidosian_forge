import uuid
from oslotest import base as test_base
from testtools import matchers
import webob
import webob.dec
from oslo_middleware import request_id
class AltHeader(request_id.RequestId):
    compat_headers = ['x-compute-req-id', 'x-silly-id']