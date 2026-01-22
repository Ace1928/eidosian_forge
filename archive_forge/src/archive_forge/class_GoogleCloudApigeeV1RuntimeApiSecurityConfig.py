from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeApiSecurityConfig(_messages.Message):
    """Runtime configuration for the API Security add-on.

  Fields:
    enabled: If the API Security is enabled or not.
  """
    enabled = _messages.BooleanField(1)