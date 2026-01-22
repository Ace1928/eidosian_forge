from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _AddTopLevelArgumentsAlpha(parser):
    """Adds top-level argument flags for the Alpha track."""
    instance_filter_group = parser.add_mutually_exclusive_group(required=True, help='Filters for selecting which instances to patch:')
    instance_filter_group.add_argument('--instance-filter', type=str, help='      Filter expression for selecting the instances to patch. Patching supports\n      the same filter mechanisms as `gcloud compute instances list`, allowing\n      one to patch specific instances by name, zone, label, or other criteria.\n      ', action=actions.DeprecationAction('--instance-filter', warn='          {flag_name} is deprecated; use individual filter flags instead. See\n          the command help text for more details.', removed=False, action='store'))
    _AddCommonInstanceFilterFlags(instance_filter_group)
    parser.add_argument('--retry', action='store_true', help='      Specifies whether to attempt to retry, within the duration window, if\n      patching initially fails. If omitted, the agent uses its default retry\n      strategy.')