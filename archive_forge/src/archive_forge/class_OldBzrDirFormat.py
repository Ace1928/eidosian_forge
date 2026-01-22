import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
class OldBzrDirFormat(bzrdir.BzrDirMetaFormat1):
    _lock_class = lockdir.LockDir

    def get_converter(self, format=None):
        return ConvertOldTestToMeta()

    @classmethod
    def get_format_string(cls):
        return b'Ancient Test Format'

    def _open(self, transport):
        return OldBzrDir(transport, self)