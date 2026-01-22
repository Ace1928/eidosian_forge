from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _GetAppendArg(arg_name, metavar, prop_name, is_dict_args):
    list_name = '--add-{}'.format(arg_name)
    list_help = 'Append the given values to the current {}.'.format(prop_name)
    dict_name = '--update-{}'.format(arg_name)
    dict_help = 'Update the given key-value pairs in the current {}.'.format(prop_name)
    return base.Argument(dict_name if is_dict_args else list_name, type=_GetArgType(is_dict_args), metavar=metavar, help=_GetArgHelp(dict_help, list_help, is_dict_args))