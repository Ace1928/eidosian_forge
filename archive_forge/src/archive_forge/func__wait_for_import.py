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
def _wait_for_import(self, image_id, retries=10):
    for i in range(0, retries):
        image = self.api_get('/v2/images/%s' % image_id).json
        if not image.get('os_glance_import_task'):
            break
        self.addDetail('Create-Import task id', ttc.text_content(image['os_glance_import_task']))
        time.sleep(1)
    self.assertIsNone(image.get('os_glance_import_task'), 'Timed out waiting for task to complete')
    return image