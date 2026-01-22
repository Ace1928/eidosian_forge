import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def _decode_stdoutdata(stdoutdata):
    """Convert data read from stdout/stderr to unicode"""
    if not isinstance(stdoutdata, bytes):
        return stdoutdata
    encoding = getattr(sys.__stdout__, 'encoding', locale.getpreferredencoding())
    if encoding is None:
        return stdoutdata.decode()
    return stdoutdata.decode(encoding)