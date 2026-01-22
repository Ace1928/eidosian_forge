import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def _test_decorator():
    i = 1
    resp = create_response_obj_with_req_id(REQUEST_ID)
    while True:
        yield (i, resp)
        i += 1