import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def _attach_log_tail(pr):
    try:
        brz_log = open(trace._get_brz_log_filename())
    except OSError as e:
        pr['BrzLogTail'] = repr(e)
        return
    try:
        lines = brz_log.readlines()
        pr['BrzLogTail'] = ''.join(lines[-40:])
    finally:
        brz_log.close()