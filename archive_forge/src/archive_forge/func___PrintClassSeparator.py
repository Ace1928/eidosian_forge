import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintClassSeparator(self):
    self.__printer()
    if not self.__printer.indent:
        self.__printer()