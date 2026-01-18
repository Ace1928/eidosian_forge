import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_directories(self):
    tree = self.make_branch_and_tree('zip')
    self.build_tree(['zip/a', 'zip/b/', 'zip/b/c', 'zip/d/'])
    tree.add(['a', 'b', 'b/c', 'd'])
    tree.commit('init')
    os.chdir('zip')
    self.run_bzr('export test.zip')
    zfile = zipfile.ZipFile('test.zip')
    names = sorted(zfile.namelist())
    self.assertEqual(['test/a', 'test/b/', 'test/b/c', 'test/d/'], names)
    file_attr = stat.S_IFREG | zip.FILE_PERMISSIONS
    dir_attr = stat.S_IFDIR | zip.ZIP_DIRECTORY_BIT | zip.DIR_PERMISSIONS
    a_info = zfile.getinfo(names[0])
    self.assertEqual(file_attr, a_info.external_attr)
    b_info = zfile.getinfo(names[1])
    self.assertEqual(dir_attr, b_info.external_attr)
    c_info = zfile.getinfo(names[2])
    self.assertEqual(file_attr, c_info.external_attr)
    d_info = zfile.getinfo(names[3])
    self.assertEqual(dir_attr, d_info.external_attr)