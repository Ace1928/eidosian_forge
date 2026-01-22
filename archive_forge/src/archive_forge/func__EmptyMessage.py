import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _EmptyMessage(message_type):
    return not any((message_type.enum_types, message_type.message_types, message_type.fields))