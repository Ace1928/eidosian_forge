import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _FetchCustomMappings(descriptor_ls):
    """Find and return all custom mappings for descriptors in descriptor_ls."""
    custom_mappings = []
    for descriptor in descriptor_ls:
        if isinstance(descriptor, ExtendedEnumDescriptor):
            custom_mappings.extend((_FormatCustomJsonMapping('Enum', m, descriptor) for m in descriptor.enum_mappings))
        elif isinstance(descriptor, ExtendedMessageDescriptor):
            custom_mappings.extend((_FormatCustomJsonMapping('Field', m, descriptor) for m in descriptor.field_mappings))
            custom_mappings.extend(_FetchCustomMappings(descriptor.enum_types))
            custom_mappings.extend(_FetchCustomMappings(descriptor.message_types))
    return custom_mappings