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
def test_unicode_user(self):
    ue = osutils.get_user_encoding()
    uni_val, env_val = tests.probe_unicode_in_user_encoding()
    if uni_val is None:
        raise tests.TestSkipped('Cannot find a unicode character that works in encoding %s' % (osutils.get_user_encoding(),))
    uni_username = 'jrandom' + uni_val
    encoded_username = uni_username.encode(ue)
    self.overrideEnv(self.envvar_to_override(), uni_username)
    self.assertEqual(uni_username, osutils.getuser_unicode())