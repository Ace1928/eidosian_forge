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
def _GetEffectiveApiEndpoint(api_name, api_version, client_class=None):
    """Returns effective endpoint for given api."""
    try:
        endpoint_override = properties.VALUES.api_endpoint_overrides.Property(api_name).Get()
    except properties.NoSuchPropertyError:
        endpoint_override = None
    api_def = GetApiDef(api_name, api_version)
    if api_def.apitools:
        client_class = client_class or _GetClientClass(api_name, api_version)
    else:
        client_class = client_class or _GetGapicClientClass(api_name, api_version)
    client_base_url = _GetBaseUrlFromApi(api_name, api_version)
    if endpoint_override:
        address = _BuildEndpointOverride(endpoint_override, client_base_url)
    elif _MtlsEnabled(api_name, api_version):
        address = UniversifyAddress(_GetMtlsEndpoint(api_name, api_version, client_class))
    else:
        address = client_base_url
    return address