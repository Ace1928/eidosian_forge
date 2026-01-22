from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Output only. The name of this instance is in the form of
      projects/{project}/locations/{location}/instances/{instance}.
    updateMask: Field mask is used to specify the fields that the update will
      overwrite in an instance resource. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask, all the supported fields (labels and options currently) will be
      overwritten.
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)