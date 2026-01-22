from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersPatchRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersPatchRequest
  object.

  Fields:
    allowMissing: If set to true, and the bare metal standalone cluster is not
      found, the request will create a new bare metal standalone cluster with
      the provided configuration. The user must have both create and update
      permission to call Update with allow_missing set to true.
    bareMetalStandaloneCluster: A BareMetalStandaloneCluster resource to be
      passed as the request body.
    name: Immutable. The bare metal standalone cluster resource name.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the BareMetalStandaloneCluster resource by the update.
      The fields specified in the update_mask are relative to the resource,
      not the full request. A field will be overwritten if it is in the mask.
      If the user does not provide a mask then all populated fields in the
      BareMetalStandaloneCluster message will be updated. Empty fields will be
      ignored unless a field mask is used.
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    bareMetalStandaloneCluster = _messages.MessageField('BareMetalStandaloneCluster', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)