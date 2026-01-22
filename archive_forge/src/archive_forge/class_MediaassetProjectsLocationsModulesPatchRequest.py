from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsModulesPatchRequest(_messages.Message):
    """A MediaassetProjectsLocationsModulesPatchRequest object.

  Fields:
    module: A Module resource to be passed as the request body.
    name: The resource name of the module, in the following form:
      `projects/{project}/locations/{location}/module/{module}`.
    updateMask: Field mask is used to specify the fields to be overwritten in
      the Module resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then all fields will be overwritten.
  """
    module = _messages.MessageField('Module', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)