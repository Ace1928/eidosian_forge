from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareAdminClustersPatchRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareAdminClustersPatchRequest object.

  Fields:
    name: Immutable. The VMware admin cluster resource name.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the VMwareAdminCluster resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all populated fields in the
      VmwareAdminCluster message will be updated. Empty fields will be ignored
      unless a field mask is used.
    validateOnly: Validate the request without actually doing any updates.
    vmwareAdminCluster: A VmwareAdminCluster resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)
    vmwareAdminCluster = _messages.MessageField('VmwareAdminCluster', 4)