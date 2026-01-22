from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
def _UpdateMessageHelper(message, diff):
    for key, val in six.iteritems(diff):
        if hasattr(message, key):
            if isinstance(val, dict):
                _UpdateMessageHelper(getattr(message, key), diff[key])
            else:
                setattr(message, key, val)
    return message