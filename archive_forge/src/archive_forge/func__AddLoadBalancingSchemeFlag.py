from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _AddLoadBalancingSchemeFlag(parser):
    """Add --load-balancing-scheme flag."""
    help_text = "  Specifies the load balancer type this validation request is for. Use\n  `EXTERNAL_MANAGED` for global external Application Load Balancer. Use\n  `EXTERNAL` for classic Application Load Balancer.\n\n  Other load balancer types are not supported. For more information, refer to\n  [Choosing a load balancer](https://cloud.google.com/load-balancing/docs/choosing-load-balancer/).\n\n  If unspecified, the load balancing scheme will be inferred from the backend\n  service resources this URL map references. If that can not be inferred (for\n  example, this URL map only references backend buckets, or this URL map is\n  for rewrites and redirects only and doesn't reference any backends),\n  `EXTERNAL` will be used as the default type.\n\n  If specified, the scheme must not conflict with the load balancing\n  scheme of the backend service resources this URL map references.\n  "
    parser.add_argument('--load-balancing-scheme', choices=['EXTERNAL', 'EXTERNAL_MANAGED'], help=help_text, required=False)