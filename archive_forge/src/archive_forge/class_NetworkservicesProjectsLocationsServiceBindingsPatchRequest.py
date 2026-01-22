from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsServiceBindingsPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsServiceBindingsPatchRequest object.

  Fields:
    name: Required. Name of the ServiceBinding resource. It matches pattern
      `projects/*/locations/global/serviceBindings/service_binding_name`.
    serviceBinding: A ServiceBinding resource to be passed as the request
      body.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the ServiceBinding resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    serviceBinding = _messages.MessageField('ServiceBinding', 2)
    updateMask = _messages.StringField(3)