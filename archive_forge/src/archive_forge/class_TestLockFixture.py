import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
class TestLockFixture(test_base.BaseTestCase):

    def setUp(self):
        super(TestLockFixture, self).setUp()
        self.config = self.useFixture(config.Config(lockutils.CONF)).config
        self.tempdir = tempfile.mkdtemp()

    def _check_in_lock(self):
        self.assertTrue(self.lock.exists())

    def tearDown(self):
        self._check_in_lock()
        super(TestLockFixture, self).tearDown()

    def test_lock_fixture(self):
        self.config(lock_path=self.tempdir, group='oslo_concurrency')
        fixture = fixtures.LockFixture('test-lock')
        self.useFixture(fixture)
        self.lock = fixture.lock