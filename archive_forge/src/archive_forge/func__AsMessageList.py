import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _AsMessageList(msg):
    """Convert the provided list-as-JsonValue to a list."""
    from apitools.base.py import extra_types

    def _IsRepeatedJsonValue(msg):
        """Return True if msg is a repeated value as a JsonValue."""
        if isinstance(msg, extra_types.JsonArray):
            return True
        if isinstance(msg, extra_types.JsonValue) and msg.array_value:
            return True
        return False
    if not _IsRepeatedJsonValue(msg):
        raise ValueError('invalid argument to _AsMessageList')
    if isinstance(msg, extra_types.JsonValue):
        msg = msg.array_value
    if isinstance(msg, extra_types.JsonArray):
        msg = msg.entries
    return msg