import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def _PrintMessages(proto_printer, message_list):
    message_list = sorted(message_list, key=operator.attrgetter('name'))
    for message_type in message_list:
        proto_printer.PrintMessage(message_type)