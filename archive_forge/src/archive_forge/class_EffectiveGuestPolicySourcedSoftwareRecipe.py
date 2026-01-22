from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EffectiveGuestPolicySourcedSoftwareRecipe(_messages.Message):
    """A guest policy recipe including its source.

  Fields:
    softwareRecipe: A software recipe to configure on the VM instance.
    source: Name of the guest policy providing this config.
  """
    softwareRecipe = _messages.MessageField('SoftwareRecipe', 1)
    source = _messages.StringField(2)