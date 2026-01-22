from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceMoveRequest(_messages.Message):
    """A InstanceMoveRequest object.

  Fields:
    destinationZone: The URL of the destination zone to move the instance.
      This can be a full or partial URL. For example, the following are all
      valid URLs to a zone: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone -
      projects/project/zones/zone - zones/zone
    targetInstance: The URL of the target instance to move. This can be a full
      or partial URL. For example, the following are all valid URLs to an
      instance: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /instances/instance - projects/project/zones/zone/instances/instance -
      zones/zone/instances/instance
  """
    destinationZone = _messages.StringField(1)
    targetInstance = _messages.StringField(2)