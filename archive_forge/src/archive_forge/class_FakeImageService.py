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
class FakeImageService(object):

    def __init__(self, http_conn, authtoken):
        self.authtoken = authtoken

    def get_images(self):
        if self.authtoken == 'livesourcetoken':
            return FAKEIMAGES_LIVEMASTER
        return FAKEIMAGES

    def get_image(self, id):
        return FakeHttpResponse({}, b'data')

    def get_image_meta(self, id):
        for img in FAKEIMAGES:
            if img['id'] == id:
                return img
        return {}

    def add_image_meta(self, meta):
        return ({'status': http.OK}, None)

    def add_image(self, meta, data):
        return ({'status': http.OK}, None)