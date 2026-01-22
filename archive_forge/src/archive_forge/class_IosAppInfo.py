from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IosAppInfo(_messages.Message):
    """iOS app information

  Fields:
    name: The name of the app. Required
  """
    name = _messages.StringField(1)