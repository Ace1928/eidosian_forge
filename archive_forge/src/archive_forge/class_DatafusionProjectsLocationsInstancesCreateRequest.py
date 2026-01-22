from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. The name of the instance to create. Instance name
      can only contain lowercase alphanumeric characters and hyphens. It must
      start with a letter and must not end with a hyphen. It can have a
      maximum of 30 characters.
    parent: Required. The instance's project and location in the format
      projects/{project}/locations/{location}.
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)