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
def test_rest_get_image(self):
    c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
    image_contents = 'THISISTHEIMAGEBODY'
    c.conn.prime_request('GET', 'v1/images/%s' % IMG_RESPONSE_ACTIVE['id'], '', {'x-auth-token': 'noauth'}, http.OK, image_contents, IMG_RESPONSE_ACTIVE)
    body = c.get_image(IMG_RESPONSE_ACTIVE['id'])
    self.assertEqual(image_contents, body.read())