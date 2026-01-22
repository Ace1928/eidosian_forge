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
def AddSnoozeSettingsFlags(parser, update=False):
    """Adds snooze settings flags to the parser."""
    snooze_settings_group = parser.add_group(help='      Snooze Settings.\n      If any of these are specified, they will overwrite fields in the\n      `--snooze-from-file` flags if specified.')
    AddDisplayNameFlag(snooze_settings_group, resource='Snooze')
    if not update:
        AddCriteriaPoliciesFlag(snooze_settings_group, resource='Snooze')
    AddStartTimeFlag(snooze_settings_group, resource='Snooze')
    AddEndTimeFlag(snooze_settings_group, resource='Snooze')