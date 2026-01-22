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
class AbsolutePathError(FilterError):

    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        super().__init__(f'member {tarinfo.name!r} has an absolute path')