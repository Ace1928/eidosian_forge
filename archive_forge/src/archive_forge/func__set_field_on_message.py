import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def _set_field_on_message(msg, key, value):
    """Set helper for protobuf Messages."""
    if isinstance(value, (collections.abc.MutableSequence, tuple)):
        while getattr(msg, key):
            getattr(msg, key).pop()
        for item in value:
            if isinstance(item, collections.abc.Mapping):
                getattr(msg, key).add(**item)
            else:
                getattr(msg, key).extend([item])
    elif isinstance(value, collections.abc.Mapping):
        for item_key, item_value in value.items():
            set(getattr(msg, key), item_key, item_value)
    elif isinstance(value, message.Message):
        getattr(msg, key).CopyFrom(value)
    else:
        setattr(msg, key, value)