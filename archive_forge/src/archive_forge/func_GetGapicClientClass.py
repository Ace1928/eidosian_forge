from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetGapicClientClass(api_name, api_version, transport=apis_util.GapicTransport.GRPC):
    """Returns the GAPIC client class for the API specified in the args.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    transport: apis_util.GapicTransport, The transport class to obtain.

  Raises:
    GapicRestUnsupportedError: If transport is REST.

  Returns:
    The specified GAPIC API Client class.
  """
    if transport == apis_util.GapicTransport.REST:
        raise GapicRestUnsupportedError()
    return apis_internal._GetGapicClientClass(api_name, api_version, transport_choice=transport)