from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagersRecreateRequest(_messages.Message):
    """A RegionInstanceGroupManagersRecreateRequest object.

  Fields:
    instances: The URLs of one or more instances to recreate. This can be a
      full URL or a partial URL, such as
      zones/[ZONE]/instances/[INSTANCE_NAME].
  """
    instances = _messages.StringField(1, repeated=True)