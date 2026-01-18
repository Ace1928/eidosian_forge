import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
Create a field mask by comparing two messages.

    Args:
        original (~google.protobuf.message.Message): the original message.
            If set to None, this field will be interpreted as an empty
            message.
        modified (~google.protobuf.message.Message): the modified message.
            If set to None, this field will be interpreted as an empty
            message.

    Returns:
        google.protobuf.field_mask_pb2.FieldMask: field mask that contains
        the list of field names that have different values between the two
        messages. If the messages are equivalent, then the field mask is empty.

    Raises:
        ValueError: If the ``original`` or ``modified`` are not the same type.
    