import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestBackupNames(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.backups = []

    def backup_exists(self, name):
        return name in self.backups

    def available_backup_name(self, name):
        backup_name = osutils.available_backup_name(name, self.backup_exists)
        self.backups.append(backup_name)
        return backup_name

    def assertBackupName(self, expected, name):
        self.assertEqual(expected, self.available_backup_name(name))

    def test_empty(self):
        self.assertBackupName('file.~1~', 'file')

    def test_existing(self):
        self.available_backup_name('file')
        self.available_backup_name('file')
        self.assertBackupName('file.~3~', 'file')
        self.backups.remove('file.~2~')
        self.assertBackupName('file.~2~', 'file')