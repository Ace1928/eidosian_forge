from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2AndroidKeyRestrictions(_messages.Message):
    """The Android apps that are allowed to use the key.

  Fields:
    allowedApplications: A list of Android applications that are allowed to
      make API calls with this key.
  """
    allowedApplications = _messages.MessageField('V2AndroidApplication', 1, repeated=True)