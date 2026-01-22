from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetentionValue(_messages.Message):
    """A collection of object level retention parameters.

    Fields:
      mode: The bucket's object retention mode, can only be Unlocked or
        Locked.
      retainUntilTime: A time in RFC 3339 format until which object retention
        protects this object.
    """
    mode = _messages.StringField(1)
    retainUntilTime = _message_types.DateTimeField(2)