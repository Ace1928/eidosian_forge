import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ProtoPrinter(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for proto printers."""

    @abc.abstractmethod
    def PrintPreamble(self, package, version, file_descriptor):
        """Print the file docstring and import lines."""

    @abc.abstractmethod
    def PrintEnum(self, enum_type):
        """Print the given enum declaration."""

    @abc.abstractmethod
    def PrintMessage(self, message_type):
        """Print the given message declaration."""