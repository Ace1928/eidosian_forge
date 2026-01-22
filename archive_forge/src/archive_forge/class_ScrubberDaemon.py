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
class ScrubberDaemon(Server):
    """
    Server object that starts/stops/manages the Scrubber server
    """

    def __init__(self, test_dir, policy_file, daemon=False, **kwargs):
        super(ScrubberDaemon, self).__init__(test_dir, 0)
        self.server_name = 'scrubber'
        self.server_module = 'glance.cmd.%s' % self.server_name
        self.daemon = daemon
        self.image_dir = os.path.join(self.test_dir, 'images')
        self.scrub_time = 5
        self.pid_file = os.path.join(self.test_dir, 'scrubber.pid')
        self.log_file = os.path.join(self.test_dir, 'scrubber.log')
        self.metadata_encryption_key = '012345678901234567890123456789ab'
        self.lock_path = self.test_dir
        default_sql_connection = SQLITE_CONN_TEMPLATE % self.test_dir
        self.sql_connection = os.environ.get('GLANCE_TEST_SQL_CONNECTION', default_sql_connection)
        self.policy_file = policy_file
        self.policy_default_rule = 'default'
        self.conf_base = '[DEFAULT]\ndebug = %(debug)s\nlog_file = %(log_file)s\ndaemon = %(daemon)s\nwakeup_time = 2\nscrub_time = %(scrub_time)s\nmetadata_encryption_key = %(metadata_encryption_key)s\nlock_path = %(lock_path)s\nsql_idle_timeout = 3600\n[database]\nconnection = %(sql_connection)s\n[glance_store]\nfilesystem_store_datadir=%(image_dir)s\n[oslo_policy]\npolicy_file = %(policy_file)s\npolicy_default_rule = %(policy_default_rule)s\n'

    def start(self, expect_exit=True, expected_exitcode=0, **kwargs):
        if 'daemon' in kwargs:
            expect_exit = False
        return super(ScrubberDaemon, self).start(expect_exit=expect_exit, expected_exitcode=expected_exitcode, **kwargs)