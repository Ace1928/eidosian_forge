from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
def _ReplaceOne(c):
    """Returns the homoglyph or escaped replacement for c."""
    equiv = homoglyphs.get(c)
    if equiv is not None:
        return equiv
    try:
        c.encode('ascii')
        return c
    except UnicodeError:
        pass
    try:
        return c.encode('unicode-escape').decode('ascii')
    except UnicodeError:
        return '?'