import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _PrintFields(fields, printer):
    for extended_field in fields:
        field = extended_field.field_descriptor
        printed_field_info = {'name': field.name, 'module': '_messages', 'type_name': '', 'type_format': '', 'number': field.number, 'label_format': '', 'variant_format': '', 'default_format': ''}
        message_field = _MESSAGE_FIELD_MAP.get(field.type_name)
        if message_field:
            printed_field_info['module'] = '_message_types'
            field_type = message_field
        elif field.type_name == 'extra_types.DateField':
            printed_field_info['module'] = 'extra_types'
            field_type = extra_types.DateField
        else:
            field_type = messages.Field.lookup_field_type_by_variant(field.variant)
        if field_type in (messages.EnumField, messages.MessageField):
            printed_field_info['type_format'] = "'%s', " % field.type_name
        if field.label == protorpc_descriptor.FieldDescriptor.Label.REQUIRED:
            printed_field_info['label_format'] = ', required=True'
        elif field.label == protorpc_descriptor.FieldDescriptor.Label.REPEATED:
            printed_field_info['label_format'] = ', repeated=True'
        if field_type.DEFAULT_VARIANT != field.variant:
            printed_field_info['variant_format'] = ', variant=_messages.Variant.%s' % field.variant
        if field.default_value:
            if field_type in [messages.BytesField, messages.StringField]:
                default_value = repr(field.default_value)
            elif field_type is messages.EnumField:
                try:
                    default_value = str(int(field.default_value))
                except ValueError:
                    default_value = repr(field.default_value)
            else:
                default_value = field.default_value
            printed_field_info['default_format'] = ', default=%s' % (default_value,)
        printed_field_info['type_name'] = field_type.__name__
        args = ''.join(('%%(%s)s' % field for field in ('type_format', 'number', 'label_format', 'variant_format', 'default_format')))
        format_str = '%%(name)s = %%(module)s.%%(type_name)s(%s)' % args
        printer(format_str % printed_field_info)