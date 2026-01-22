import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def _create_and_upload(self, data_iter=None, expected_code=204, visibility=None):
    data = {'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw'}
    if visibility:
        data['visibility'] = visibility
    resp = self.api_post('/v2/images', json=data)
    self.assertEqual(201, resp.status_code, resp.text)
    image = jsonutils.loads(resp.text)
    if data_iter:
        resp = self.api_put('/v2/images/%s/file' % image['id'], headers={'Content-Type': 'application/octet-stream'}, body_file=data_iter)
    else:
        resp = self.api_put('/v2/images/%s/file' % image['id'], headers={'Content-Type': 'application/octet-stream'}, data=b'IMAGEDATA')
    self.assertEqual(expected_code, resp.status_code)
    return image['id']