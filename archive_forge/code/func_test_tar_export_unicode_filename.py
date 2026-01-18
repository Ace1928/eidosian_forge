import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_tar_export_unicode_filename(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('tar')
    fname = '€.txt'
    self.build_tree(['tar/' + fname])
    tree.add([fname])
    tree.commit('first')
    self.run_bzr('export test.tar -d tar')
    with tarfile.open('test.tar') as ball:
        self.assertEqual(['test/' + fname], [osutils.safe_unicode(n) for n in ball.getnames()])