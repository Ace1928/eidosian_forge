from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
def _AddCommonArgsForDryRunCreate(parser, prefix='', version='v1'):
    """Adds arguments common to the two dry-run create modes.

  Args:
    parser: The argparse parser to add the arguments to.
    prefix: Optional prefix, e.g. 'perimeter-' to use for the argument names.
    version: Api version. e.g. v1alpha, v1beta, v1.
  """
    parser.add_argument('--{}resources'.format(prefix), metavar='resources', type=arg_parsers.ArgList(), default=None, help='Comma-separated list of resources (currently only projects, in the\n              form `projects/<projectnumber>`) in this perimeter.')
    parser.add_argument('--{}restricted-services'.format(prefix), metavar='restricted_services', type=arg_parsers.ArgList(), default=None, help='Comma-separated list of services to which the perimeter boundary\n              *does* apply (for example, `storage.googleapis.com`).')
    parser.add_argument('--{}access-levels'.format(prefix), metavar='access_levels', type=arg_parsers.ArgList(), default=None, help='Comma-separated list of IDs for access levels (in the same policy)\n              that an intra-perimeter request must satisfy to be allowed.')
    vpc_group = parser.add_argument_group()
    vpc_group.add_argument('--{}enable-vpc-accessible-services'.format(prefix), action='store_true', default=None, help='Whether to restrict API calls within the perimeter to those in the\n              `vpc-allowed-services` list.')
    vpc_group.add_argument('--{}vpc-allowed-services'.format(prefix), metavar='vpc_allowed_services', type=arg_parsers.ArgList(), default=None, help='Comma-separated list of APIs accessible from within the Service\n              Perimeter. In order to include all restricted services, use\n              reference "RESTRICTED-SERVICES". Requires vpc-accessible-services\n              be enabled.')
    parser.add_argument('--{}ingress-policies'.format(prefix), metavar='YAML_FILE', type=perimeters.ParseIngressPolicies(version), default=None, help='Path to a file containing a list of Ingress Policies.\n              This file contains a list of YAML-compliant objects representing\n              Ingress Policies described in the API reference.\n              For more information about the alpha version, see:\n              https://cloud.google.com/access-context-manager/docs/reference/rest/v1alpha/accessPolicies.servicePerimeters\n              For more information about non-alpha versions, see:\n              https://cloud.google.com/access-context-manager/docs/reference/rest/v1/accessPolicies.servicePerimeters')
    parser.add_argument('--{}egress-policies'.format(prefix), metavar='YAML_FILE', type=perimeters.ParseEgressPolicies(version), default=None, help='Path to a file containing a list of Egress Policies.\n              This file contains a list of YAML-compliant objects representing\n              Egress Policies described in the API reference.\n              For more information about the alpha version, see:\n              https://cloud.google.com/access-context-manager/docs/reference/rest/v1alpha/accessPolicies.servicePerimeters\n              For more information about non-alpha versions, see:\n              https://cloud.google.com/access-context-manager/docs/reference/rest/v1/accessPolicies.servicePerimeters')