from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSImage(_messages.Message):
    """Operation System image.

  Fields:
    applicableInstanceTypes: Instance types this image is applicable to.
      [Available types](https://cloud.google.com/bare-metal/docs/bms-
      planning#server_configurations)
    code: OS Image code.
    description: OS Image description.
    name: Output only. OS Image's unique name.
    supportedNetworkTemplates: Network templates that can be used with this OS
      Image.
  """
    applicableInstanceTypes = _messages.StringField(1, repeated=True)
    code = _messages.StringField(2)
    description = _messages.StringField(3)
    name = _messages.StringField(4)
    supportedNetworkTemplates = _messages.StringField(5, repeated=True)