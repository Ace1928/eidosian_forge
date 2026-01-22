import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def assertTarANameAndContent(self, ball, root=''):
    fname = root + 'a'
    ball_iter = iter(ball)
    tar_info = next(ball_iter)
    self.assertEqual(fname, tar_info.name)
    self.assertEqual(tarfile.REGTYPE, tar_info.type)
    self.assertEqual(len(self._file_content), tar_info.size)
    f = ball.extractfile(tar_info)
    if self._file_content != f.read():
        self.fail('File content has been corrupted. Check that all streams are handled in binary mode.')
    self.assertRaises(StopIteration, next, ball_iter)