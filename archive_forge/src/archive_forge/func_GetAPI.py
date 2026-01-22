from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetAPI(api_name, api_version=None):
    """Get a specific API definition.

  Args:
    api_name: str, The name of the API.
    api_version: str, The version string of the API.

  Returns:
    API, The API definition.
  """
    api_version = _ValidateAndGetDefaultVersion(api_name, api_version)
    api_def = apis_internal.GetApiDef(api_name, api_version)
    if api_def.apitools:
        api_client = apis_internal._GetClientClassFromDef(api_def)
    else:
        api_client = apis_internal._GetGapicClientClass(api_name, api_version)
    if hasattr(api_client, 'BASE_URL'):
        base_url = api_client.BASE_URL
    else:
        try:
            base_url = apis_internal._GetResourceModule(api_name, api_version).BASE_URL
        except ImportError:
            base_url = 'https://{}.googleapis.com/{}'.format(api_name, api_version)
    return API(api_name, api_version, api_def.default_version, api_client, base_url)