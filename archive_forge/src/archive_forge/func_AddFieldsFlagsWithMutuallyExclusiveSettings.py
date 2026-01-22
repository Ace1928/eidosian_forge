from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddFieldsFlagsWithMutuallyExclusiveSettings(parser, fields_help, add_settings_func, fields_choices=None, **kwargs):
    """Adds fields flags with mutually excludisve settings."""
    update_group = parser.add_group(mutex=True)
    update_group.add_argument('--fields', metavar='field', type=arg_parsers.ArgList(choices=fields_choices), help=fields_help)
    add_settings_func(update_group, **kwargs)