import os
import re
import tempfile
from os_win import _utils
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def _get_raw_icacls_info(self, path):
    return _utils.execute('icacls.exe', path)[0]