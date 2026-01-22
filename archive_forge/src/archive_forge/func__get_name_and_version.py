from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def _get_name_and_version(name, version, for_filename=False):
    """Return the distribution name with version.

    If for_filename is true, return a filename-escaped form."""
    if for_filename:
        name = _FILESAFE.sub('-', name)
        version = _FILESAFE.sub('-', version.replace(' ', '.'))
    return '%s-%s' % (name, version)