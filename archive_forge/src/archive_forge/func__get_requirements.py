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
def _get_requirements(self, req_attr):
    md = self.metadata
    reqts = getattr(md, req_attr)
    logger.debug('%s: got requirements %r from metadata: %r', self.name, req_attr, reqts)
    return set(md.get_requirements(reqts, extras=self.extras, env=self.context))