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
class ReplicationUtilitiesTestCase(test_utils.BaseTestCase):

    def test_check_upload_response_headers(self):
        glance_replicator._check_upload_response_headers({'status': 'active'}, None)
        d = {'image': {'status': 'active'}}
        glance_replicator._check_upload_response_headers({}, jsonutils.dumps(d))
        self.assertRaises(exception.UploadException, glance_replicator._check_upload_response_headers, {}, None)

    def test_image_present(self):
        client = FakeImageService(None, 'noauth')
        self.assertTrue(glance_replicator._image_present(client, '5dcddce0-cba5-4f18-9cf4-9853c7b207a6'))
        self.assertFalse(glance_replicator._image_present(client, uuid.uuid4()))

    def test_dict_diff(self):
        a = {'a': 1, 'b': 2, 'c': 3}
        b = {'a': 1, 'b': 2}
        c = {'a': 1, 'b': 1, 'c': 3}
        d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        self.assertFalse(glance_replicator._dict_diff(a, a))
        self.assertTrue(glance_replicator._dict_diff(a, b))
        self.assertTrue(glance_replicator._dict_diff(a, c))
        self.assertFalse(glance_replicator._dict_diff(a, d))