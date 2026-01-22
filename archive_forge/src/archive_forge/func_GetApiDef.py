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
def GetApiDef(api_name, api_version):
    """Returns the APIDef for the specified API and version.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.

  Raises:
    apis_util.UnknownAPIError: If api_name does not exist in the APIs map.
    apis_util.UnknownVersionError: If api_version does not exist for given
      api_name in the APIs map.

  Returns:
    APIDef, The APIDef for the specified API and version.
  """
    api_name, api_name_alias = _GetApiNameAndAlias(api_name)
    if api_name not in apis_map.MAP:
        raise apis_util.UnknownAPIError(api_name)
    version_overrides = properties.VALUES.api_client_overrides.AllValues()
    version_override = version_overrides.get('{}/{}'.format(api_name, api_version))
    if not version_override:
        version_override = version_overrides.get(api_name_alias, None)
    api_version = version_override or api_version
    api_versions = apis_map.MAP[api_name]
    if api_version is None or api_version not in api_versions:
        raise apis_util.UnknownVersionError(api_name, api_version)
    else:
        api_def = api_versions[api_version]
    return api_def