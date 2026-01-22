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
class FakeAuthMiddleware(wsgi.Middleware):

    def __init__(self, app, is_admin=False):
        super(FakeAuthMiddleware, self).__init__(app)
        self.is_admin = is_admin

    def process_request(self, req):
        auth_token = req.headers.get('X-Auth-Token')
        user = None
        tenant = None
        roles = []
        if auth_token:
            user, tenant, role = auth_token.split(':')
            if tenant.lower() == 'none':
                tenant = None
            roles = [role]
            req.headers['X-User-Id'] = user
            req.headers['X-Tenant-Id'] = tenant
            req.headers['X-Roles'] = role
            req.headers['X-Identity-Status'] = 'Confirmed'
        kwargs = {'user': user, 'tenant': tenant, 'roles': roles, 'is_admin': self.is_admin, 'auth_token': auth_token}
        req.context = context.RequestContext(**kwargs)