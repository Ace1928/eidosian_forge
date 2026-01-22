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
def _AddCommonInstanceFilterFlags(mutually_exclusive_group):
    """Adds instance filter flags to a mutually exclusive argument group."""
    mutually_exclusive_group.add_argument('--instance-filter-all', action='store_true', help='A filter that targets all instances in the project.')
    individual_filters_group = mutually_exclusive_group.add_group(help='    Individual filters. The targeted instances must meet all criteria specified.\n    ')
    individual_filters_group.add_argument('--instance-filter-group-labels', action='append', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='      A filter that represents a label set. Targeted instances must have all\n      specified labels in this set. For example, "env=prod and app=web".\n\n      This flag can be repeated. Targeted instances must have at least one of\n      these label sets. This allows targeting of disparate groups, for example,\n      "(env=prod and app=web) or (env=staging and app=web)".')
    individual_filters_group.add_argument('--instance-filter-zones', metavar='INSTANCE_FILTER_ZONES', type=arg_parsers.ArgList(), help='      A filter that targets instances in any of the specified zones. Leave empty\n      to target instances in any zone.')
    individual_filters_group.add_argument('--instance-filter-names', metavar='INSTANCE_FILTER_NAMES', type=arg_parsers.ArgList(), help='      A filter that targets instances of any of the specified names. Instances\n      are specified by the URI in the form\n      "zones/<ZONE>/instances/<INSTANCE_NAME>",\n      "projects/<PROJECT_ID>/zones/<ZONE>/instances/<INSTANCE_NAME>", or\n      "https://www.googleapis.com/compute/v1/projects/<PROJECT_ID>/zones/<ZONE>/instances/<INSTANCE_NAME>".\n      ')
    individual_filters_group.add_argument('--instance-filter-name-prefixes', metavar='INSTANCE_FILTER_NAME_PREFIXES', type=arg_parsers.ArgList(), help='      A filter that targets instances whose name starts with one of these\n      prefixes. For example, "prod-".')