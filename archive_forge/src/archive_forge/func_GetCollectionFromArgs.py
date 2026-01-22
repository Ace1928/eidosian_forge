from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log as sdk_log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def GetCollectionFromArgs(args, collection_suffix):
    """Returns the collection derived from args and the suffix.

  Args:
    args: command line args.
    collection_suffix: suffix of collection

  Returns:
    The collection.
  """
    if args.organization:
        prefix = 'logging.organizations'
    elif args.folder:
        prefix = 'logging.folders'
    elif args.billing_account:
        prefix = 'logging.billingAccounts'
    else:
        prefix = 'logging.projects'
    return '{0}.{1}'.format(prefix, collection_suffix)