from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScanSensitiveDataSetting(_messages.Message):
    """Scan sensitive data setting.

  Fields:
    roleNameToScanSensitiveData: Optional. AWS scanning sensitive data role
      name. This is role used to scan sensitive data under AWS accounts and
      this role is only required and used when scanning_sensitive_data_enabled
      is set to true.
    scanSensitiveDataEnabled: Optional. Whether we enable scanning sensitive
      data or not. Setting this to true means that this connection is enabled
      for SDP (Sensitive Data Protection) to scan sensitive data in customers'
      AWS accounts, which requires extra scan sensitive data related
      permissions otherwise scanning sensitive data will fail.
  """
    roleNameToScanSensitiveData = _messages.StringField(1)
    scanSensitiveDataEnabled = _messages.BooleanField(2)