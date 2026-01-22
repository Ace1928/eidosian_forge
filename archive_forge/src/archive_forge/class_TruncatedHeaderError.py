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
class TruncatedHeaderError(HeaderError):
    """Exception for truncated headers."""
    pass