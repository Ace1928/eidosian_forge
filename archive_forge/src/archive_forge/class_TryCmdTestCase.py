import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class TryCmdTestCase(test_base.BaseTestCase):

    def test_keep_warnings(self):
        self.useFixture(fixtures.MonkeyPatch('oslo_concurrency.processutils.execute', fake_execute))
        o, e = processutils.trycmd('this is a command'.split(' '))
        self.assertNotEqual('', o)
        self.assertNotEqual('', e)

    def test_keep_warnings_from_raise(self):
        self.useFixture(fixtures.MonkeyPatch('oslo_concurrency.processutils.execute', fake_execute_raises))
        o, e = processutils.trycmd('this is a command'.split(' '), discard_warnings=True)
        self.assertIsNotNone(o)
        self.assertNotEqual('', e)

    def test_discard_warnings(self):
        self.useFixture(fixtures.MonkeyPatch('oslo_concurrency.processutils.execute', fake_execute))
        o, e = processutils.trycmd('this is a command'.split(' '), discard_warnings=True)
        self.assertIsNotNone(o)
        self.assertEqual('', e)