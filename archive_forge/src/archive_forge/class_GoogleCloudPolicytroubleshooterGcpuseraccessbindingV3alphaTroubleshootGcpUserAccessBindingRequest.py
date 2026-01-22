from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingRequest(_messages.Message):
    """Request to troubleshoot GcpUserAccessBinding.

  Fields:
    troubleshootingToken: Optional. The troubleshooting token can be generated
      when customers get access denied by the GcpUserAccessBinding.
  """
    troubleshootingToken = _messages.StringField(1)