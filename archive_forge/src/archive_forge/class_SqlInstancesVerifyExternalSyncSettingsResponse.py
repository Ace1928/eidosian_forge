from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesVerifyExternalSyncSettingsResponse(_messages.Message):
    """Instance verify external sync settings response.

  Fields:
    errors: List of migration violations.
    kind: This is always `sql#migrationSettingErrorList`.
    warnings: List of migration warnings.
  """
    errors = _messages.MessageField('SqlExternalSyncSettingError', 1, repeated=True)
    kind = _messages.StringField(2)
    warnings = _messages.MessageField('SqlExternalSyncSettingError', 3, repeated=True)