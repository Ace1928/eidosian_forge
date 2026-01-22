from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2IosKeyRestrictions(_messages.Message):
    """The iOS apps that are allowed to use the key.

  Fields:
    allowedBundleIds: A list of bundle IDs that are allowed when making API
      calls with this key.
  """
    allowedBundleIds = _messages.StringField(1, repeated=True)