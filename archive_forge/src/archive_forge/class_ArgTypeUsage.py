from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
class ArgTypeUsage(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for flags types that need to provide additional usage info."""

    @property
    @abc.abstractmethod
    def hidden(self):
        """Whether the argument is hidden."""

    @abc.abstractmethod
    def GetUsageMetavar(self, is_custom_metavar, metavar):
        """Returns the metavar for flag with type self."""

    @abc.abstractmethod
    def GetUsageExample(self, shorthand):
        """Returns the example user input value for flag with type self."""

    @abc.abstractmethod
    def GetUsageHelpText(self, field_name, required, flag_name):
        """Returns the help text for flag with type self."""