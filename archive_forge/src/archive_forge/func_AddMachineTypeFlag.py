from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddMachineTypeFlag(parser, hidden=False):
    """Add a machine type flag."""
    global _machine_type_flag_map
    _machine_type_flag_map = arg_utils.ChoiceEnumMapper('--machine-type', cloudbuild_util.GetMessagesModule().BuildOptions.MachineTypeValueValuesEnum, include_filter=lambda s: six.text_type(s) != 'UNSPECIFIED', help_str='Machine type used to run the build.', hidden=hidden)
    _machine_type_flag_map.choice_arg.AddToParser(parser)