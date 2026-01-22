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
@staticmethod
def CleanName(name):
    """Perform generic name cleaning."""
    name = re.sub('[^_A-Za-z0-9]', '_', name)
    if name[0].isdigit():
        name = '_%s' % name
    while keyword.iskeyword(name) or name == 'exec':
        name = '%s_' % name
    if name.startswith('__'):
        name = 'f%s' % name
    return name