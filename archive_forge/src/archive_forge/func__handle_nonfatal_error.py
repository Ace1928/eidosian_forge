from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _handle_nonfatal_error(self, e):
    """Handle non-fatal error (ExtractError) according to errorlevel"""
    if self.errorlevel > 1:
        raise
    else:
        self._dbg(1, 'tarfile: %s' % e)