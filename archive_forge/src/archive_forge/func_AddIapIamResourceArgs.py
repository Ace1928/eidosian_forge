from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def AddIapIamResourceArgs(parser, use_region_arg=False):
    """Adds flags for an IAP IAM resource.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
    use_region_arg: Whether or not to show and accept the region argument.
  """
    group = parser.add_group()
    group.add_argument('--resource-type', required=True, choices=RESOURCE_TYPE_ENUM, help='Resource type of the IAP resource.')
    group.add_argument('--service', help='Service name.')
    if use_region_arg:
        group.add_argument('--region', help='Region name. Should only be specified with `--resource-type=backend-services`.')
    group.add_argument('--version', help='Service version. Should only be specified with `--resource-type=app-engine`.')