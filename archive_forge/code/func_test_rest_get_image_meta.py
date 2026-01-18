import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_rest_get_image_meta(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    c.conn.prime_request('HEAD', 'v1/images/%s' % IMG_RESPONSE_ACTIVE['id'], '', {'x-auth-token': 'noauth'}, http.OK, '', IMG_RESPONSE_ACTIVE)
    header = c.get_image_meta(IMG_RESPONSE_ACTIVE['id'])
    self.assertIn('id', header)