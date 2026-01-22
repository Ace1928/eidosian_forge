from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1AndroidKeyRestrictions(_messages.Message):
    """Key restrictions that are specific to android keys.

  Fields:
    allowedApplications: A list of Android applications that are allowed to
      make API calls with this key.
  """
    allowedApplications = _messages.MessageField('V2alpha1AndroidApplication', 1, repeated=True)