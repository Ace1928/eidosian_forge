from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GuestAttributesEntry(_messages.Message):
    """A guest attributes namespace/key/value entry.

  Fields:
    key: Key for the guest attribute entry.
    namespace: Namespace for the guest attribute entry.
    value: Value for the guest attribute entry.
  """
    key = _messages.StringField(1)
    namespace = _messages.StringField(2)
    value = _messages.StringField(3)