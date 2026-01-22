import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def _open_crash_file():
    crash_dir = bedding.crash_dir()
    if not osutils.isdir(crash_dir):
        os.makedirs(crash_dir, mode=384)
    date_string = time.strftime('%Y-%m-%dT%H:%M', time.gmtime())
    if sys.platform == 'win32':
        user_part = ''
    else:
        user_part = '.%d' % os.getuid()
    filename = osutils.pathjoin(crash_dir, 'brz{}.{}.crash'.format(user_part, date_string))
    return (filename, os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384), 'wb'))