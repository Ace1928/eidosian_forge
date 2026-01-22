import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
def NormalizeNumberFields(pb):
    """Normalizes types and precisions of number fields in a protocol buffer.

  Due to subtleties in the python protocol buffer implementation, it is possible
  for values to have different types and precision depending on whether they
  were set and retrieved directly or deserialized from a protobuf. This function
  normalizes integer values to ints and longs based on width, 32-bit floats to
  five digits of precision to account for python always storing them as 64-bit,
  and ensures doubles are floating point for when they're set to integers.

  Modifies pb in place. Recurses into nested objects.

  Args:
    pb: proto2 message.

  Returns:
    the given pb, modified in place.
  """
    for desc, values in pb.ListFields():
        is_repeated = True
        if desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
            is_repeated = False
            values = [values]
        normalized_values = None
        if desc.type in (descriptor.FieldDescriptor.TYPE_INT64, descriptor.FieldDescriptor.TYPE_UINT64, descriptor.FieldDescriptor.TYPE_SINT64):
            normalized_values = [int(x) for x in values]
        elif desc.type in (descriptor.FieldDescriptor.TYPE_INT32, descriptor.FieldDescriptor.TYPE_UINT32, descriptor.FieldDescriptor.TYPE_SINT32, descriptor.FieldDescriptor.TYPE_ENUM):
            normalized_values = [int(x) for x in values]
        elif desc.type == descriptor.FieldDescriptor.TYPE_FLOAT:
            normalized_values = [round(x, 6) for x in values]
        elif desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
            normalized_values = [round(float(x), 7) for x in values]
        if normalized_values is not None:
            if is_repeated:
                pb.ClearField(desc.name)
                getattr(pb, desc.name).extend(normalized_values)
            else:
                setattr(pb, desc.name, normalized_values[0])
        if desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE or desc.type == descriptor.FieldDescriptor.TYPE_GROUP:
            if desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE and desc.message_type.has_options and desc.message_type.GetOptions().map_entry:
                if desc.message_type.fields_by_number[2].type == descriptor.FieldDescriptor.TYPE_MESSAGE:
                    for v in six.itervalues(values):
                        NormalizeNumberFields(v)
            else:
                for v in values:
                    NormalizeNumberFields(v)
    return pb