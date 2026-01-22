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
class FindConfigFilesTestCase(BaseTestCase):

    def test_find_config_files(self):
        config_files = [os.path.expanduser('~/.blaa/blaa.conf'), '/etc/foo.conf']
        self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
        self.assertEqual(cfg.find_config_files(project='blaa'), config_files)

    def test_find_config_files_overrides(self):
        """Ensure priority of directories is enforced.

        Ensure we will only ever return two files: $project.conf and
        $prog.conf.
        """
        config_files = [os.path.expanduser('~/.foo/foo.conf'), os.path.expanduser('~/foo.conf'), os.path.expanduser('~/bar.conf'), '/etc/foo/foo.conf', '/etc/foo/bar.conf', '/etc/foo.conf', '/etc/bar.conf']
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
        expected = [os.path.expanduser('~/.foo/foo.conf'), os.path.expanduser('~/bar.conf')]
        actual = cfg.find_config_files(project='foo', prog='bar')
        self.assertEqual(expected, actual)

    def test_find_config_files_snap(self):
        config_files = ['/snap/nova/current/etc/blaa/blaa.conf', '/var/snap/nova/common/etc/blaa/blaa.conf']
        fake_env = {'SNAP': '/snap/nova/current/', 'SNAP_COMMON': '/var/snap/nova/common/'}
        self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
        self.useFixture(fixtures.MonkeyPatch('os.environ', fake_env))
        self.assertEqual(cfg.find_config_files(project='blaa'), ['/var/snap/nova/common/etc/blaa/blaa.conf'])

    def test_find_config_files_with_extension(self):
        config_files = ['/etc/foo.json']
        self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
        self.assertEqual(cfg.find_config_files(project='blaa'), [])
        self.assertEqual(cfg.find_config_files(project='blaa', extension='.json'), config_files)