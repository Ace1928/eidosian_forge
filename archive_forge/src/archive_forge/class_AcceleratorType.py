from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcceleratorType(_messages.Message):
    """A accelerator type that a Node can be configured with.

  Fields:
    acceleratorConfigs: The accelerator config.
    name: The resource name.
    type: The accelerator type.
  """
    acceleratorConfigs = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    name = _messages.StringField(2)
    type = _messages.StringField(3)