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
def _filter_out_stat(self, result):
    """Filter out the stat value from the walkdirs result"""
    for dirdetail, dirblock in result:
        new_dirblock = []
        for info in dirblock:
            new_dirblock.append((info[0], info[1], info[2], info[4]))
        dirblock[:] = new_dirblock