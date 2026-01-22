from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _Escape(s):
    """Return s with format special characters escaped."""
    r = []
    n = 0
    for c in s:
        if c == _ESCAPE:
            r.append(_ESCAPE + _ESCAPED_ESCAPE + _ESCAPE)
        elif c == ':':
            r.append(_ESCAPE + _ESCAPED_COLON + _ESCAPE)
        elif c == '{':
            if n > 0:
                r.append(_ESCAPE + _ESCAPED_LEFT_CURLY + _ESCAPE)
            else:
                r.append('{')
            n += 1
        elif c == '}':
            n -= 1
            if n > 0:
                r.append(_ESCAPE + _ESCAPED_RIGHT_CURLY + _ESCAPE)
            else:
                r.append('}')
        else:
            r.append(c)
    return ''.join(r)