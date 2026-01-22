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
def _GetGapicClientInstance(api_name, api_version, credentials, address_override_func=None, transport_choice=apis_util.GapicTransport.GRPC, attempt_direct_path=False):
    """Returns an instance of the GAPIC API client specified in the args.

  For apitools API clients, the API endpoint override is something like
  http://compute.googleapis.com/v1/. For GAPIC API clients, the DEFAULT_ENDPOINT
  is something like compute.googleapis.com. To use the same endpoint override
  property for both, we use the netloc of the API endpoint override.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    credentials: google.auth.credentials.Credentials, the credentials to use.
    address_override_func: function, function to call to override the client
      host. It takes a single argument which is the original host.
    transport_choice: apis_util.GapicTransport, The transport to be used by the
      client.
    attempt_direct_path: bool, True if we want to attempt direct path gRPC where
      possible.

  Returns:
    An instance of the specified GAPIC API client.
  """

    def AddressOverride(address):
        try:
            endpoint_override = properties.VALUES.api_endpoint_overrides.Property(api_name).Get()
        except properties.NoSuchPropertyError:
            endpoint_override = None
        if endpoint_override:
            address = urlparse(endpoint_override).netloc
        if address_override_func:
            address = address_override_func(address)
        if endpoint_override is not None:
            return address
        return UniversifyAddress(address)
    client_class = _GetGapicClientClass(api_name, api_version, transport_choice=transport_choice)
    return client_class(credentials, address_override_func=AddressOverride, mtls_enabled=_MtlsEnabled(api_name, api_version), attempt_direct_path=attempt_direct_path)