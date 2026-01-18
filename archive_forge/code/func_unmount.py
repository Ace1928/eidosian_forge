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
def unmount(self):
    pathname = os.path.abspath(os.path.join(self.dirname, self.filename))
    if pathname not in sys.path:
        logger.debug('%s not in path', pathname)
    else:
        sys.path.remove(pathname)
        if pathname in _hook.impure_wheels:
            _hook.remove(pathname)
        if not _hook.impure_wheels:
            if _hook in sys.meta_path:
                sys.meta_path.remove(_hook)