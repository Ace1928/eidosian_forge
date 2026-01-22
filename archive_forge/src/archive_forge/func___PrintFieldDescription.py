import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintFieldDescription(self, description):
    for line in textwrap.wrap(description, self.__printer.CalculateWidth() - 3):
        self.__printer('// %s', line)