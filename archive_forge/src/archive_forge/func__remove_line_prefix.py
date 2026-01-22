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
def _remove_line_prefix(self, value):
    if self.metadata_version in ('1.0', '1.1'):
        return _LINE_PREFIX_PRE_1_2.sub('\n', value)
    else:
        return _LINE_PREFIX_1_2.sub('\n', value)