from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesPatchRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesPatchRequest object.

  Fields:
    name: Identifier. Name of the volume
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Volume resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
    volume: A Volume resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    volume = _messages.MessageField('Volume', 3)