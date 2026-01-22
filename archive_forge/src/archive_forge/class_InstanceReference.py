from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceReference(_messages.Message):
    """A reference to a Compute Engine instance.

  Fields:
    instanceId: The unique identifier of the Compute Engine instance.
    instanceName: The user-friendly name of the Compute Engine instance.
    publicEciesKey: The public ECIES key used for sharing data with this
      instance.
    publicKey: The public RSA key used for sharing data with this instance.
  """
    instanceId = _messages.StringField(1)
    instanceName = _messages.StringField(2)
    publicEciesKey = _messages.StringField(3)
    publicKey = _messages.StringField(4)