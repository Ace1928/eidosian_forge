from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsPatchRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsPatchRequest
  object.

  Fields:
    allowMissing: If set to true, and the VMware node pool is not found, the
      request will create a new VMware node pool with the provided
      configuration. The user must have both create and update permission to
      call Update with allow_missing set to true.
    name: Immutable. The resource name of this node pool.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the VMwareNodePool resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all populated fields in the VMwareNodePool
      message will be updated. Empty fields will be ignored unless a field
      mask is used.
    validateOnly: Validate the request without actually doing any updates.
    vmwareNodePool: A VmwareNodePool resource to be passed as the request
      body.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)
    vmwareNodePool = _messages.MessageField('VmwareNodePool', 5)