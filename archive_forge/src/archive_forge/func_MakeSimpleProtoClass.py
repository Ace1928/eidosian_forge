from collections import OrderedDict
import hashlib
import os
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message_factory
def MakeSimpleProtoClass(fields, full_name=None, pool=None):
    """Create a Protobuf class whose fields are basic types.

  Note: this doesn't validate field names!

  Args:
    fields: dict of {name: field_type} mappings for each field in the proto. If
        this is an OrderedDict the order will be maintained, otherwise the
        fields will be sorted by name.
    full_name: optional str, the fully-qualified name of the proto type.
    pool: optional DescriptorPool instance.
  Returns:
    a class, the new protobuf class with a FileDescriptor.
  """
    factory = message_factory.MessageFactory(pool=pool)
    if full_name is not None:
        try:
            proto_cls = _GetMessageFromFactory(factory, full_name)
            return proto_cls
        except KeyError:
            pass
    field_items = fields.items()
    if not isinstance(fields, OrderedDict):
        field_items = sorted(field_items)
    fields_hash = hashlib.sha1()
    for f_name, f_type in field_items:
        fields_hash.update(f_name.encode('utf-8'))
        fields_hash.update(str(f_type).encode('utf-8'))
    proto_file_name = fields_hash.hexdigest() + '.proto'
    if full_name is None:
        full_name = 'net.proto2.python.public.proto_builder.AnonymousProto_' + fields_hash.hexdigest()
        try:
            proto_cls = _GetMessageFromFactory(factory, full_name)
            return proto_cls
        except KeyError:
            pass
    factory.pool.Add(_MakeFileDescriptorProto(proto_file_name, full_name, field_items))
    return _GetMessageFromFactory(factory, full_name)