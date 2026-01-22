import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _PrintEnums(proto_printer, enum_types):
    """Print all enums to the given proto_printer."""
    enum_types = sorted(enum_types, key=operator.attrgetter('name'))
    for enum_type in enum_types:
        proto_printer.PrintEnum(enum_type)