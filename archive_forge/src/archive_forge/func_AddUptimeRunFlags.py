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
def AddUptimeRunFlags(parser, update=False):
    """Adds uptime check run flags to the parser."""
    uptime_settings_group = parser.add_group(help='Settings.')
    uptime_settings_group.add_argument('--period', help='The time between uptime check or synthetic monitor executions in\n              minutes, defaults to `1`. Can be set for synthetic monitors.', choices=UPTIME_PERIODS)
    uptime_settings_group.add_argument('--timeout', help='The maximum amount of time in seconds to wait for the request to complete, defaults to `60`. Can be set for synthetic monitors.', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=60))
    if update:
        AddDisplayNameFlag(uptime_settings_group, resource='uptime check or synthetic monitor', positional=False)
        uptime_regions_group = uptime_settings_group.add_group(help='Uptime check selected regions.', mutex=True)
        uptime_regions_group.add_argument('--set-regions', metavar='region', help='The list of regions from which the check is run. At least 3 regions must be\n            selected.', type=arg_parsers.ArgList(choices=UPTIME_REGIONS))
        uptime_regions_group.add_argument('--add-regions', metavar='region', help='The list of regions to add to the uptime check.', type=arg_parsers.ArgList(choices=UPTIME_REGIONS))
        uptime_regions_group.add_argument('--remove-regions', metavar='region', help='The list of regions to remove from the uptime check.', type=arg_parsers.ArgList(choices=UPTIME_REGIONS))
        uptime_regions_group.add_argument('--clear-regions', help='Clear all regions on the uptime check. This setting acts the same as if all available\n            regions were selected.', type=bool)
    else:
        uptime_settings_group.add_argument('--regions', metavar='field', help='The list of regions from which the check is run. At least 3 regions must be selected.\n            Defaults to all available regions.', type=arg_parsers.ArgList(choices=UPTIME_REGIONS))
    if update:
        AddUpdateLabelsFlags('user-labels', uptime_settings_group, 'User labels. Can be set for synthetic monitors.')
    else:
        AddCreateLabelsFlag(uptime_settings_group, 'user-labels', 'User labels. Can be set for synthetic monitors.', skip_extra_message=True)