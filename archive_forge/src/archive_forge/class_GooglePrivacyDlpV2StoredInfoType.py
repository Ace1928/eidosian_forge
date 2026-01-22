from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StoredInfoType(_messages.Message):
    """StoredInfoType resource message that contains information about the
  current version and any pending updates.

  Fields:
    currentVersion: Current version of the stored info type.
    name: Resource name.
    pendingVersions: Pending versions of the stored info type. Empty if no
      versions are pending.
  """
    currentVersion = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeVersion', 1)
    name = _messages.StringField(2)
    pendingVersions = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeVersion', 3, repeated=True)