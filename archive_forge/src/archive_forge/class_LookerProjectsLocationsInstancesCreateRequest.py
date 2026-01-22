from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. The unique instance identifier. Must contain only
      lowercase letters, numbers, or hyphens, with the first character a
      letter and the last a letter or a number. 63 characters maximum.
    parent: Required. Format: `projects/{project}/locations/{location}`.
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)