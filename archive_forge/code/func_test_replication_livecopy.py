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
def test_replication_livecopy(self):
    options = collections.UserDict()
    options.chunksize = 4096
    options.dontreplicate = 'dontrepl dontreplabsent'
    options.sourcetoken = 'livesourcetoken'
    options.targettoken = 'livetargettoken'
    options.metaonly = False
    args = ['localhost:9292', 'localhost:9393']
    orig_img_service = glance_replicator.get_image_service
    try:
        glance_replicator.get_image_service = get_image_service
        updated = glance_replicator.replication_livecopy(options, args)
    finally:
        glance_replicator.get_image_service = orig_img_service
    self.assertEqual(2, len(updated))