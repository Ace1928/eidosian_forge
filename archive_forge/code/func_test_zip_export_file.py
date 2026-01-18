import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_file(self):
    tree = self.make_basic_tree()
    self.run_bzr('export -d tree test.zip')
    self.assertZipANameAndContent(zipfile.ZipFile('test.zip'), root='test/')