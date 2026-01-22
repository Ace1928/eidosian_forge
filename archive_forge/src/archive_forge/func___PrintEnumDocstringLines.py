import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintEnumDocstringLines(self, enum_type):
    description = enum_type.description or '%s enum type.' % enum_type.name
    for line in textwrap.wrap('r"""%s' % description, self.__printer.CalculateWidth()):
        self.__printer(line)
    PrintIndentedDescriptions(self.__printer, enum_type.values, 'Values')
    self.__printer('"""')