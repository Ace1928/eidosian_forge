from __future__ import unicode_literals
import base64
import codecs
import datetime
from email import message_from_file
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile
from . import __version__, DistlibException
from .compat import sysconfig, ZipFile, fsdecode, text_type, filter
from .database import InstalledDistribution
from .metadata import Metadata, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME
from .util import (FileOperator, convert_path, CSVReader, CSVWriter, Cache,
from .version import NormalizedVersion, UnsupportedVersionError
def _derive_abi():
    parts = ['cp', VER_SUFFIX]
    if sysconfig.get_config_var('Py_DEBUG'):
        parts.append('d')
    if IMP_PREFIX == 'cp':
        vi = sys.version_info[:2]
        if vi < (3, 8):
            wpm = sysconfig.get_config_var('WITH_PYMALLOC')
            if wpm is None:
                wpm = True
            if wpm:
                parts.append('m')
            if vi < (3, 3):
                us = sysconfig.get_config_var('Py_UNICODE_SIZE')
                if us == 4 or (us is None and sys.maxunicode == 1114111):
                    parts.append('u')
    return ''.join(parts)