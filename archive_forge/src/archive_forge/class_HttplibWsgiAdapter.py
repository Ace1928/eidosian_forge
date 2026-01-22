import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
class HttplibWsgiAdapter(object):

    def __init__(self, app):
        self.app = app
        self.req = None

    def request(self, method, url, body=None, headers=None):
        if headers is None:
            headers = {}
        self.req = webob.Request.blank(url, method=method, headers=headers)
        self.req.body = body

    def getresponse(self):
        response = self.req.get_response(self.app)
        return FakeHTTPResponse(response.status_code, response.headers, response.body)