from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def AddUpdateMapFlags(parser, flag_name, long_name=None, key_type=None, value_type=None):
    """Add flags for updating values of a map-of-atomic-values property.

  Args:
    parser: The argument parser
    flag_name: The name for the property to be used in flag names
    long_name: The name for the property to be used in help text
    key_type: A function to apply to map keys.
    value_type: A function to apply to map values.
  """
    if not long_name:
        long_name = flag_name
    group = parser.add_mutually_exclusive_group()
    update_remove_group = group.add_argument_group(help='Only --update-{0} and --remove-{0} can be used together.  If both are specified, --remove-{0} will be applied first.'.format(flag_name))
    AddMapUpdateFlag(update_remove_group, flag_name, long_name, key_type=key_type, value_type=value_type)
    AddMapRemoveFlag(update_remove_group, flag_name, long_name, key_type=key_type)
    AddMapClearFlag(group, flag_name, long_name)
    AddMapSetFlag(group, flag_name, long_name, key_type=key_type, value_type=value_type)
    AddMapSetFileFlag(group, flag_name, long_name, key_type=key_type, value_type=value_type)