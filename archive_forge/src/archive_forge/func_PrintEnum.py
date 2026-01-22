import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def PrintEnum(self, enum_type):
    self.__printer('class %s(_messages.Enum):', enum_type.name)
    with self.__printer.Indent():
        self.__PrintEnumDocstringLines(enum_type)
        enum_values = sorted(enum_type.values, key=operator.attrgetter('number'))
        for enum_value in enum_values:
            self.__printer('%s = %s', enum_value.name, enum_value.number)
        if not enum_type.values:
            self.__printer('pass')
    self.__PrintClassSeparator()