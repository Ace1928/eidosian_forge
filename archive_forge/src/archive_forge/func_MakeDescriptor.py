import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def MakeDescriptor(desc_proto, package='', build_file_if_cpp=True, syntax=None, edition=None, file_desc=None):
    """Make a protobuf Descriptor given a DescriptorProto protobuf.

  Handles nested descriptors. Note that this is limited to the scope of defining
  a message inside of another message. Composite fields can currently only be
  resolved if the message is defined in the same scope as the field.

  Args:
    desc_proto: The descriptor_pb2.DescriptorProto protobuf message.
    package: Optional package name for the new message Descriptor (string).
    build_file_if_cpp: Update the C++ descriptor pool if api matches. Set to
      False on recursion, so no duplicates are created.
    syntax: The syntax/semantics that should be used.  Set to "proto3" to get
      proto3 field presence semantics.
    edition: The edition that should be used if syntax is "edition".
    file_desc: A FileDescriptor to place this descriptor into.

  Returns:
    A Descriptor for protobuf messages.
  """
    from google.protobuf import descriptor_pb2
    proto_name = binascii.hexlify(os.urandom(16)).decode('ascii')
    if package:
        file_name = os.path.join(package.replace('.', '/'), proto_name + '.proto')
    else:
        file_name = proto_name + '.proto'
    if api_implementation.Type() != 'python' and build_file_if_cpp:
        file_descriptor_proto = descriptor_pb2.FileDescriptorProto()
        file_descriptor_proto.message_type.add().MergeFrom(desc_proto)
        if package:
            file_descriptor_proto.package = package
        file_descriptor_proto.name = file_name
        _message.default_pool.Add(file_descriptor_proto)
        result = _message.default_pool.FindFileByName(file_descriptor_proto.name)
        if _USE_C_DESCRIPTORS:
            return result.message_types_by_name[desc_proto.name]
    if file_desc is None:
        file_desc = FileDescriptor(pool=None, name=file_name, package=package, syntax=syntax, edition=edition, options=None, serialized_pb='', dependencies=[], public_dependencies=[], create_key=_internal_create_key)
    full_message_name = [desc_proto.name]
    if package:
        full_message_name.insert(0, package)
    enum_types = {}
    for enum_proto in desc_proto.enum_type:
        full_name = '.'.join(full_message_name + [enum_proto.name])
        enum_desc = EnumDescriptor(enum_proto.name, full_name, None, [EnumValueDescriptor(enum_val.name, ii, enum_val.number, create_key=_internal_create_key) for ii, enum_val in enumerate(enum_proto.value)], file=file_desc, create_key=_internal_create_key)
        enum_types[full_name] = enum_desc
    nested_types = {}
    for nested_proto in desc_proto.nested_type:
        full_name = '.'.join(full_message_name + [nested_proto.name])
        nested_desc = MakeDescriptor(nested_proto, package='.'.join(full_message_name), build_file_if_cpp=False, syntax=syntax, edition=edition, file_desc=file_desc)
        nested_types[full_name] = nested_desc
    fields = []
    for field_proto in desc_proto.field:
        full_name = '.'.join(full_message_name + [field_proto.name])
        enum_desc = None
        nested_desc = None
        if field_proto.json_name:
            json_name = field_proto.json_name
        else:
            json_name = None
        if field_proto.HasField('type_name'):
            type_name = field_proto.type_name
            full_type_name = '.'.join(full_message_name + [type_name[type_name.rfind('.') + 1:]])
            if full_type_name in nested_types:
                nested_desc = nested_types[full_type_name]
            elif full_type_name in enum_types:
                enum_desc = enum_types[full_type_name]
        field = FieldDescriptor(field_proto.name, full_name, field_proto.number - 1, field_proto.number, field_proto.type, FieldDescriptor.ProtoTypeToCppProtoType(field_proto.type), field_proto.label, None, nested_desc, enum_desc, None, False, None, options=_OptionsOrNone(field_proto), has_default_value=False, json_name=json_name, file=file_desc, create_key=_internal_create_key)
        fields.append(field)
    desc_name = '.'.join(full_message_name)
    return Descriptor(desc_proto.name, desc_name, None, None, fields, list(nested_types.values()), list(enum_types.values()), [], options=_OptionsOrNone(desc_proto), file=file_desc, create_key=_internal_create_key)