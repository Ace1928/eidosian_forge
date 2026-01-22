from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddRulesArg(parser):
    parser.add_argument('--rules', help=textwrap.dedent('          Path to YAML file containing NAT Rules applied to the NAT.\n          The YAML file format must follow the REST API schema for NAT Rules.\n          See [API Discovery docs](https://www.googleapis.com/discovery/v1/apis/compute/alpha/rest)\n          for reference.'), required=False)