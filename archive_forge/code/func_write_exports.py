from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def write_exports(self, exports):
    """
        Write a dictionary of exports to a file in .ini format.
        :param exports: A dictionary of exports, mapping an export category to
                        a list of :class:`ExportEntry` instances describing the
                        individual export entries.
        """
    rf = self.get_distinfo_file(EXPORTS_FILENAME)
    with open(rf, 'w') as f:
        write_exports(exports, f)