from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoclassValue(_messages.Message):
    """The bucket's Autoclass configuration.

    Fields:
      enabled: Whether or not Autoclass is enabled on this bucket
      toggleTime: A date and time in RFC 3339 format representing the instant
        at which "enabled" was last toggled.
    """
    enabled = _messages.BooleanField(1)
    toggleTime = _message_types.DateTimeField(2)