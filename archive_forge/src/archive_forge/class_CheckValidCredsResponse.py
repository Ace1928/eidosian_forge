from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckValidCredsResponse(_messages.Message):
    """A response indicating whether the credentials exist and are valid.

  Fields:
    hasValidCreds: If set to `true`, the credentials exist and are valid.
  """
    hasValidCreds = _messages.BooleanField(1)