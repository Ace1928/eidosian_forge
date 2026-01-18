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
def test_replication_dump(self):
    tempdir = self.useFixture(fixtures.TempDir()).path
    options = collections.UserDict()
    options.chunksize = 4096
    options.sourcetoken = 'sourcetoken'
    options.metaonly = False
    args = ['localhost:9292', tempdir]
    orig_img_service = glance_replicator.get_image_service
    self.addCleanup(setattr, glance_replicator, 'get_image_service', orig_img_service)
    glance_replicator.get_image_service = get_image_service
    glance_replicator.replication_dump(options, args)
    for active in ['5dcddce0-cba5-4f18-9cf4-9853c7b207a6', '37ff82db-afca-48c7-ae0b-ddc7cf83e3db']:
        imgfile = os.path.join(tempdir, active)
        self.assertTrue(os.path.exists(imgfile))
        self.assertTrue(os.path.exists('%s.img' % imgfile))
        with open(imgfile) as f:
            d = jsonutils.loads(f.read())
            self.assertIn('status', d)
            self.assertIn('id', d)
            self.assertIn('size', d)
    for inactive in ['f4da1d2a-40e8-4710-b3aa-0222a4cc887b']:
        imgfile = os.path.join(tempdir, inactive)
        self.assertTrue(os.path.exists(imgfile))
        self.assertFalse(os.path.exists('%s.img' % imgfile))
        with open(imgfile) as f:
            d = jsonutils.loads(f.read())
            self.assertIn('status', d)
            self.assertIn('id', d)
            self.assertIn('size', d)