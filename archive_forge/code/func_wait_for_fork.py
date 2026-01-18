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
def wait_for_fork(pid, raise_error=True, expected_exitcode=0, force=True):
    """
    Wait for a process to complete

    This function will wait for the given pid to complete.  If the
    exit code does not match that of the expected_exitcode an error
    is raised.
    """
    term_timer = timeutils.StopWatch(5)
    term_timer.start()
    nice_timer = timeutils.StopWatch(7)
    nice_timer.start()
    total_timer = timeutils.StopWatch(10)
    total_timer.start()
    while not total_timer.expired():
        try:
            cpid, rc = os.waitpid(pid, force and os.WNOHANG or 0)
            if cpid == 0 and force:
                if not term_timer.expired():
                    pass
                elif not nice_timer.expired():
                    LOG.warning('Killing child %i with SIGTERM', pid)
                    os.kill(pid, signal.SIGTERM)
                else:
                    LOG.warning('Killing child %i with SIGKILL', pid)
                    os.kill(pid, signal.SIGKILL)
                    expected_exitcode = signal.SIGKILL
                time.sleep(1)
                continue
            LOG.info('waitpid(%i) returned %i,%i', pid, cpid, rc)
            if rc != expected_exitcode:
                raise RuntimeError('The exit code %d is not %d' % (rc, expected_exitcode))
            return rc
        except ChildProcessError:
            return 0
        except Exception as e:
            LOG.error('Got wait error: %s', e)
            if raise_error:
                raise
    raise RuntimeError('Gave up waiting for %i to exit!' % pid)