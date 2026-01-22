from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddConnectManagedActiveDirectoryArg(parser):
    """Adds a --managed-ad flag to the parser.

  Args:
    parser: argparse parser.
  """
    managed_ad_arg_spec = {'domain': str, 'computer': str}
    managed_ad_help = '        Managed Active Directory configuration for an instance. Specifies both\n        the domain name and a computer name (unique to the domain) to be created\n        by the filestore instance.\n\n         domain\n            The desired domain full uri. i.e:\n            projects/PROJECT/locations/global/domains/DOMAIN\n\n         computer\n            The desired active directory computer name to be created by\n            the filestore instance when connecting to the domain.\n  '
    parser.add_argument('--managed-ad', type=arg_parsers.ArgDict(spec=managed_ad_arg_spec, required_keys=['domain', 'computer']), required=False, help=managed_ad_help)