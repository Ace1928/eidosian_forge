from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateResourceValueConfigRequest(_messages.Message):
    """Request message to create single resource value config

  Fields:
    parent: Required. Resource name of the new ResourceValueConfig's parent.
    resourceValueConfig: Required. The resource value config being created.
  """
    parent = _messages.StringField(1)
    resourceValueConfig = _messages.MessageField('GoogleCloudSecuritycenterV2ResourceValueConfig', 2)