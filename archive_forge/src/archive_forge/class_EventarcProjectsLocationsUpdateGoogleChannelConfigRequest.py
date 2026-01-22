from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsUpdateGoogleChannelConfigRequest(_messages.Message):
    """A EventarcProjectsLocationsUpdateGoogleChannelConfigRequest object.

  Fields:
    googleChannelConfig: A GoogleChannelConfig resource to be passed as the
      request body.
    name: Required. The resource name of the config. Must be in the format of,
      `projects/{project}/locations/{location}/googleChannelConfig`.
    updateMask: The fields to be updated; only fields explicitly provided are
      updated. If no field mask is provided, all provided fields in the
      request are updated. To update all fields, provide a field mask of "*".
  """
    googleChannelConfig = _messages.MessageField('GoogleChannelConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)