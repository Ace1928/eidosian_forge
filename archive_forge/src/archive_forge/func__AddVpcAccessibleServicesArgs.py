from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _AddVpcAccessibleServicesArgs(parser, list_help, enable_help):
    """Add to the parser arguments for this service restriction type."""
    group = parser.add_argument_group()
    repeated.AddPrimitiveArgs(group, 'perimeter', 'vpc-allowed-services', 'vpc allowed services', metavar='VPC_SERVICE', include_set=False, additional_help=list_help)
    group.add_argument('--enable-vpc-accessible-services', default=None, action='store_true', help=enable_help)