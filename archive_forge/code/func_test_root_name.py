import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
def test_root_name(self):
    self.assertEqual('mytest', get_root_name('../mytest.tar'))
    self.assertEqual('mytar', get_root_name('mytar.tar'))
    self.assertEqual('mytar', get_root_name('mytar.tar.bz2'))
    self.assertEqual('tar.tar.tar', get_root_name('tar.tar.tar.tgz'))
    self.assertEqual('bzr-0.0.5', get_root_name('bzr-0.0.5.tar.gz'))
    self.assertEqual('bzr-0.0.5', get_root_name('bzr-0.0.5.zip'))
    self.assertEqual('bzr-0.0.5', get_root_name('bzr-0.0.5'))
    self.assertEqual('mytar', get_root_name('a/long/path/mytar.tgz'))
    self.assertEqual('other', get_root_name('../parent/../dir/other.tbz2'))
    self.assertEqual('', get_root_name('-'))