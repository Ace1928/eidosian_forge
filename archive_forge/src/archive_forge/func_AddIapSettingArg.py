from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def AddIapSettingArg(parser, use_region_arg=False):
    """Adds flags for an IAP settings resource.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
    use_region_arg: Whether or not to show and accept the region argument.
  """
    group = parser.add_group()
    group.add_argument('--organization', help='Organization ID.')
    group.add_argument('--folder', help='Folder ID.')
    group.add_argument('--project', help='Project ID.')
    group.add_argument('--resource-type', choices=SETTING_RESOURCE_TYPE_ENUM, help='Resource type of the IAP resource.')
    group.add_argument('--service', help="Service name. Optional when ``resource-type'' is ``compute'' or ``app-engine''.")
    if use_region_arg:
        group.add_argument('--region', help="Region name. Not applicable for ``app-engine''. Optional when ``resource-type'' is ``compute''.")
    group.add_argument('--version', help="Version name. Not applicable for ``compute''. Optional when ``resource-type'' is ``app-engine''.")