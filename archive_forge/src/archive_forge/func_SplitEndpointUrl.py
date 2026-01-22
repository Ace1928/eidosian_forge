from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def SplitEndpointUrl(url):
    """Returns api_name, api_version, resource_path tuple for an API URL.

  Supports the following formats:
  # Google API production/staging endpoints.
  http(s)://www.googleapis.com/{api}/{version}/{resource-path}
  http(s)://stagingdomain/{api}/{version}/{resource-path}
  http(s)://{api}.googleapis.com/{api}/{version}/{resource-path}
  http(s)://{api}.googleapis.com/apis/{kube-api.name}/{version}/{resource-path}
  http(s)://{api}.googleapis.com/{version}/{resource-path}
  http(s)://googledomain/{api}
  # Non-Google API endpoints.
  http(s)://someotherdomain/{api}/{version}/{resource-path}

  Args:
    url: str, The resource url.

  Returns:
    (str, str, str): The API name, version, resource_path.
    For a malformed URL, the return value for API name is undefined, version is
    None, and resource path is ''.

  Raises: InvalidEndpointException.
  """
    tokens = _StripUrl(url).split('/')
    version_index = _GetApiVersionIndex(tokens)
    domain = tokens[0]
    if version_index < 1:
        return (_ExtractApiNameFromDomain(domain), None, '')
    version = tokens[version_index]
    resource_path = '/'.join(tokens[version_index + 1:])
    if version_index == 1:
        return (_ExtractApiNameFromDomain(domain), version, resource_path)
    if version_index > 1:
        api_name = _FindKubernetesApiName(domain) or tokens[version_index - 1]
        return (api_name, version, resource_path)
    raise InvalidEndpointException(url)