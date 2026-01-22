from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
class ScalarTypeMismatchError(DecodeError):
    """Incicates a scalar property was provided a value of an unexpected type."""