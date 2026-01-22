import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _SetFieldType(self, field_proto, field_desc, package, scope):
    """Sets the field's type, cpp_type, message_type and enum_type.

    Args:
      field_proto: Data about the field in proto format.
      field_desc: The descriptor to modify.
      package: The package the field's container is in.
      scope: Enclosing scope of available types.
    """
    if field_proto.type_name:
        desc = self._GetTypeFromScope(package, field_proto.type_name, scope)
    else:
        desc = None
    if not field_proto.HasField('type'):
        if isinstance(desc, descriptor.Descriptor):
            field_proto.type = descriptor.FieldDescriptor.TYPE_MESSAGE
        else:
            field_proto.type = descriptor.FieldDescriptor.TYPE_ENUM
    field_desc.cpp_type = descriptor.FieldDescriptor.ProtoTypeToCppProtoType(field_proto.type)
    if field_proto.type == descriptor.FieldDescriptor.TYPE_MESSAGE or field_proto.type == descriptor.FieldDescriptor.TYPE_GROUP:
        field_desc.message_type = desc
    if field_proto.type == descriptor.FieldDescriptor.TYPE_ENUM:
        field_desc.enum_type = desc
    if field_proto.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        field_desc.has_default_value = False
        field_desc.default_value = []
    elif field_proto.HasField('default_value'):
        field_desc.has_default_value = True
        if field_proto.type == descriptor.FieldDescriptor.TYPE_DOUBLE or field_proto.type == descriptor.FieldDescriptor.TYPE_FLOAT:
            field_desc.default_value = float(field_proto.default_value)
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_STRING:
            field_desc.default_value = field_proto.default_value
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_BOOL:
            field_desc.default_value = field_proto.default_value.lower() == 'true'
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_ENUM:
            field_desc.default_value = field_desc.enum_type.values_by_name[field_proto.default_value].number
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_BYTES:
            field_desc.default_value = text_encoding.CUnescape(field_proto.default_value)
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
            field_desc.default_value = None
        else:
            field_desc.default_value = int(field_proto.default_value)
    else:
        field_desc.has_default_value = False
        if field_proto.type == descriptor.FieldDescriptor.TYPE_DOUBLE or field_proto.type == descriptor.FieldDescriptor.TYPE_FLOAT:
            field_desc.default_value = 0.0
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_STRING:
            field_desc.default_value = u''
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_BOOL:
            field_desc.default_value = False
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_ENUM:
            field_desc.default_value = field_desc.enum_type.values[0].number
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_BYTES:
            field_desc.default_value = b''
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
            field_desc.default_value = None
        elif field_proto.type == descriptor.FieldDescriptor.TYPE_GROUP:
            field_desc.default_value = None
        else:
            field_desc.default_value = 0
    field_desc.type = field_proto.type