from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1MutatePartnerPermissionsRequest(_messages.Message):
    """Request for updating permission settings for a partner workload.

  Fields:
    etag: Optional. The etag of the workload. If this is provided, it must
      match the server's etag.
    partnerPermissions: Required. The partner permissions to be updated.
    updateMask: Required. The list of fields to be updated. E.g. update_mask {
      paths: "partner_permissions.data_logs_viewer"}
  """
    etag = _messages.StringField(1)
    partnerPermissions = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadPartnerPermissions', 2)
    updateMask = _messages.StringField(3)