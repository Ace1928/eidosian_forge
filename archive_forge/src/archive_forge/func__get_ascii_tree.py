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
def _get_ascii_tree(self):
    tree = ['0file', '1dir/', '1dir/0file', '1dir/1dir/', '2file']
    expected_dirblocks = [((b'', '.'), [(b'0file', b'0file', 'file', './0file'), (b'1dir', b'1dir', 'directory', './1dir'), (b'2file', b'2file', 'file', './2file')]), ((b'1dir', './1dir'), [(b'1dir/0file', b'0file', 'file', './1dir/0file'), (b'1dir/1dir', b'1dir', 'directory', './1dir/1dir')]), ((b'1dir/1dir', './1dir/1dir'), [])]
    return (tree, expected_dirblocks)