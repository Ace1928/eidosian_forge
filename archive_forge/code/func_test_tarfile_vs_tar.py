import unittest
import os
import sys
import tarfile
from os.path import splitdrive
import warnings
from distutils import archive_util
from distutils.archive_util import (check_archive_formats, make_tarball,
from distutils.spawn import find_executable, spawn
from distutils.tests import support
from test.support import patch
from test.support.os_helper import change_cwd
from test.support.warnings_helper import check_warnings
@unittest.skipUnless(find_executable('tar') and find_executable('gzip') and ZLIB_SUPPORT, 'Need the tar, gzip and zlib command to run')
def test_tarfile_vs_tar(self):
    tmpdir = self._create_files()
    tmpdir2 = self.mkdtemp()
    base_name = os.path.join(tmpdir2, 'archive')
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        make_tarball(base_name, 'dist')
    finally:
        os.chdir(old_dir)
    tarball = base_name + '.tar.gz'
    self.assertTrue(os.path.exists(tarball))
    tarball2 = os.path.join(tmpdir, 'archive2.tar.gz')
    tar_cmd = ['tar', '-cf', 'archive2.tar', 'dist']
    gzip_cmd = ['gzip', '-f', '-9', 'archive2.tar']
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        spawn(tar_cmd)
        spawn(gzip_cmd)
    finally:
        os.chdir(old_dir)
    self.assertTrue(os.path.exists(tarball2))
    self.assertEqual(self._tarinfo(tarball), self._created_files)
    self.assertEqual(self._tarinfo(tarball2), self._created_files)
    base_name = os.path.join(tmpdir2, 'archive')
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        make_tarball(base_name, 'dist', compress=None)
    finally:
        os.chdir(old_dir)
    tarball = base_name + '.tar'
    self.assertTrue(os.path.exists(tarball))
    base_name = os.path.join(tmpdir2, 'archive')
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        make_tarball(base_name, 'dist', compress=None, dry_run=True)
    finally:
        os.chdir(old_dir)
    tarball = base_name + '.tar'
    self.assertTrue(os.path.exists(tarball))