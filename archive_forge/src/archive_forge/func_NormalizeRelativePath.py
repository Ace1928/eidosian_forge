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
def NormalizeRelativePath(path):
    """Normalize camelCase entries in path."""
    path_components = path.split('/')
    normalized_components = []
    for component in path_components:
        if re.match('{[A-Za-z0-9_]+}$', component):
            normalized_components.append('{%s}' % Names.CleanName(component[1:-1]))
        else:
            normalized_components.append(component)
    return '/'.join(normalized_components)