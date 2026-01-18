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
def test_rest_dict_to_headers(self):
    i = {'banana': 42, 'gerkin': 12, 'properties': {'frog': 1, 'kernel_id': None}}
    o = glance_replicator.ImageService._dict_to_headers(i)
    self.assertIn('x-image-meta-banana', o)
    self.assertIn('x-image-meta-gerkin', o)
    self.assertIn('x-image-meta-property-frog', o)
    self.assertIn('x-image-meta-property-kernel_id', o)
    self.assertEqual(o['x-image-meta-property-kernel_id'], '')
    self.assertNotIn('properties', o)