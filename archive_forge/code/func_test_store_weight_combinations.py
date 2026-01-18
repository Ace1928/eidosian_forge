import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def test_store_weight_combinations(self):
    self.start_server()
    image_id = self._create_and_import(stores=['store1', 'store2', 'store3'])
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('store1,store2,store3', image['stores'])
    self.config(weight=200, group='store2')
    self.config(weight=100, group='store3')
    self.config(weight=50, group='store1')
    self.start_server()
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('store2,store3,store1', image['stores'])
    self.config(weight=20, group='store2')
    self.config(weight=100, group='store3')
    self.config(weight=50, group='store1')
    self.start_server()
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('store3,store1,store2', image['stores'])