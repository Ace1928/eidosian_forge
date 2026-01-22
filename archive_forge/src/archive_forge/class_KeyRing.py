from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyRing(_messages.Message):
    """A KeyRing is a toplevel logical grouping of CryptoKeys.

  Fields:
    createTime: Output only. The time at which this KeyRing was created.
    name: Output only. The resource name for the KeyRing in the format
      `projects/*/locations/*/keyRings/*`.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)