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
def xattr_writes_supported(path):
    """
    Returns True if the we can write a file to the supplied
    path and subsequently write a xattr to that file.
    """
    try:
        import xattr
    except ImportError:
        return False

    def set_xattr(path, key, value):
        xattr.setxattr(path, 'user.%s' % key, value)
    fake_filepath = os.path.join(path, 'testing-checkme')
    result = True
    with open(fake_filepath, 'wb') as fake_file:
        fake_file.write(b'XXX')
        fake_file.flush()
    try:
        set_xattr(fake_filepath, 'hits', b'1')
    except IOError as e:
        if e.errno == errno.EOPNOTSUPP:
            result = False
    else:
        if os.path.exists(fake_filepath):
            os.unlink(fake_filepath)
    return result