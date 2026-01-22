from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.generated_clients.apis import apis_map
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.parse import urlparse
def _GetMtlsEndpoint(api_name, api_version, client_class=None):
    """Returns mtls endpoint."""
    api_def = GetApiDef(api_name, api_version)
    if api_def.apitools:
        client_class = client_class or _GetClientClass(api_name, api_version)
    else:
        client_class = client_class or _GetGapicClientClass(api_name, api_version)
    return api_def.mtls_endpoint_override or client_class.MTLS_BASE_URL