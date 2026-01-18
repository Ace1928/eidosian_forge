import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_walkdirs_encoding_error(self):
    self.requireFeature(features.ByteStringNamedFilesystem)
    tree = ['.bzr', '0file', '1dir/', '1dir/0file', '1dir/1dir/', '1file']
    self.build_tree(tree)
    os.rename(b'./1file', b'\xe8file')
    if b'\xe8file' not in os.listdir('.'):
        self.skipTest('Lack filesystem that preserves arbitrary bytes')
    self._save_platform_info()

    def attempt():
        for dirdetail, dirblock in osutils.walkdirs(b'.', codecs.utf_8_decode):
            pass
    self.assertRaises(UnicodeDecodeError, attempt)