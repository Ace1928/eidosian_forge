import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_tar_export_unicode_basedir(self):
    """Test for bug #413406"""
    self.requireFeature(features.UnicodeFilenameFeature)
    basedir = '€'
    os.mkdir(basedir)
    self.run_bzr(['init', basedir])
    self.run_bzr(['export', '--format', 'tgz', 'test.tar.gz', '-d', basedir])