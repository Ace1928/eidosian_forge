from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ArgFallthrough(_FallthroughBase):
    """Gets an attribute from the argparse parsed values for that arg."""

    def __init__(self, arg_name, plural=False):
        """Initializes a fallthrough for the argument associated with the attribute.

    Args:
      arg_name: str, the name of the flag or positional.
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
        resource argument is plural (i.e. parses to a list).
    """
        super(ArgFallthrough, self).__init__('provide the argument `{}` on the command line'.format(arg_name), active=True, plural=plural)
        self.arg_name = arg_name

    def _Call(self, parsed_args):
        arg_value = getattr(parsed_args, util.NamespaceFormat(self.arg_name), None)
        return arg_value

    def _Pluralize(self, value):
        if not self.plural:
            if isinstance(value, list):
                return value[0] if value else None
            return value
        if value and (not isinstance(value, list)):
            return [value]
        return value if value else []

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return other.arg_name == self.arg_name

    def __hash__(self):
        return hash(self.arg_name)