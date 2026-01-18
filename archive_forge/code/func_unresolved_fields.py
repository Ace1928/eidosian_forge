import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
@property
def unresolved_fields(self):
    """Return fields with referencing message types as strings."""
    for proto_plus_message in self.messages.values():
        for field in proto_plus_message._meta.fields.values():
            if field.message and isinstance(field.message, str) or (field.enum and isinstance(field.enum, str)):
                yield field