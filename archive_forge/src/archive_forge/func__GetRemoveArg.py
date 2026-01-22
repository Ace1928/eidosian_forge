from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _GetRemoveArg(arg_name, metavar, prop_name, is_dict_args):
    list_help = 'Remove the given values from the current {}.'.format(prop_name)
    dict_help = 'Remove the key-value pairs from the current {} with the given keys.'.format(prop_name)
    return base.Argument('--remove-{}'.format(arg_name), metavar=metavar, type=_GetArgType(is_dict_args), help=_GetArgHelp(dict_help, list_help, is_dict_args))