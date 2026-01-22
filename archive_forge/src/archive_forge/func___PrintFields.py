import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintFields(self, fields):
    for extended_field in fields:
        field = extended_field.field_descriptor
        field_type = messages.Field.lookup_field_type_by_variant(field.variant)
        self.__printer()
        self.__PrintFieldDescription(extended_field.description)
        label = str(field.label).lower()
        if field_type in (messages.EnumField, messages.MessageField):
            proto_type = field.type_name
        else:
            proto_type = str(field.variant).lower()
        default_statement = ''
        if field.default_value:
            if field_type in [messages.BytesField, messages.StringField]:
                default_value = '"%s"' % field.default_value
            elif field_type is messages.BooleanField:
                default_value = str(field.default_value).lower()
            else:
                default_value = str(field.default_value)
            default_statement = ' [default = %s]' % default_value
        self.__printer('%s %s %s = %d%s;', label, proto_type, field.name, field.number, default_statement)