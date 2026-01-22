from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class _FallthroughBase(object, metaclass=abc.ABCMeta):
    """Represents a way to get information about a concept's attribute.

  Specific implementations of Fallthrough objects must implement the method:

    _Call():
      Get a value from information given to the fallthrough.

  GetValue() is used by the Deps object to attempt to find the value of an
  attribute. The hint property is used to provide an informative error when an
  attribute can't be found.
  """

    def __init__(self, hint, active=False, plural=False):
        """Initializes a fallthrough to an arbitrary function.

    Args:
      hint: str | list[str], The user-facing message for the fallthrough
        when it cannot be resolved.
      active: bool, True if the fallthrough is considered to be "actively"
        specified, i.e. on the command line.
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
        resource argument is plural (i.e. parses to a list).
    """
        self._hint = hint
        self.active = active
        self.plural = plural

    def GetValue(self, parsed_args):
        """Gets a value from information given to the fallthrough.

    Args:
      parsed_args: the argparse namespace.

    Raises:
      FallthroughNotFoundError: If the attribute is not found.

    Returns:
      The value of the attribute.
    """
        value = self._Call(parsed_args)
        if value:
            return self._Pluralize(value)
        raise FallthroughNotFoundError()

    @abc.abstractmethod
    def _Call(self, parsed_args):
        pass

    def _Pluralize(self, value):
        """Pluralize the result of calling the fallthrough. May be overridden."""
        if not self.plural or isinstance(value, list):
            return value
        return [value] if value else []

    @property
    def hint(self):
        """String representation of the fallthrough for user-facing messaging."""
        return self._hint

    def __hash__(self):
        return hash(self.hint) + hash(self.active)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.hint == self.hint and (other.active == self.active) and (other.plural == self.plural)