import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class FindFileTestCase(BaseTestCase):

    def test_find_file_without_init(self):
        self.assertRaises(cfg.NotInitializedError, self.conf.find_file, 'foo.json')

    def test_find_policy_file(self):
        policy_file = '/etc/policy.json'
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p == policy_file))
        self.conf([])
        self.assertIsNone(self.conf.find_file('foo.json'))
        self.assertEqual(policy_file, self.conf.find_file('policy.json'))

    def test_find_policy_file_with_config_file(self):
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        paths = self.create_tempfiles([(os.path.join(dir, 'test.conf'), '[DEFAULT]'), (os.path.join(dir, 'policy.json'), '{}')], ext='')
        self.conf(['--config-file', paths[0]])
        self.assertEqual(paths[1], self.conf.find_file('policy.json'))

    def test_find_policy_file_with_multiple_config_dirs(self):
        dir1 = tempfile.mkdtemp()
        self.tempdirs.append(dir1)
        dir2 = tempfile.mkdtemp()
        self.tempdirs.append(dir2)
        self.conf(['--config-dir', dir1, '--config-dir', dir2])
        self.assertEqual(2, len(self.conf.config_dirs))
        self.assertEqual(dir1, self.conf.config_dirs[0])
        self.assertEqual(dir2, self.conf.config_dirs[1])

    def test_config_dirs_empty_list_when_nothing_parsed(self):
        self.assertEqual([], self.conf.config_dirs)

    def test_find_policy_file_with_config_dir(self):
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        dir2 = tempfile.mkdtemp()
        self.tempdirs.append(dir2)
        path = self.create_tempfiles([(os.path.join(dir, 'policy.json'), '{}')], ext='')[0]
        self.conf(['--config-dir', dir, '--config-dir', dir2])
        self.assertEqual(path, self.conf.find_file('policy.json'))