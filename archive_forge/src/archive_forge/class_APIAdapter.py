from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
class APIAdapter(object):
    """Handles making api requests in a version-agnostic way."""
    _HTTP_ERROR_FORMAT = 'HTTP request failed with status code {}. Response content: {}'

    def __init__(self, registry, client, messages, api_version):
        self.registry = registry
        self.client = client
        self.messages = messages
        self.api_version = api_version

    def _ManifestResponse(self, client, messages, option):
        return getattr(client.projects_locations_memberships.GenerateConnectManifest(messages.GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest(imagePullSecretContent=six.ensure_binary(option.image_pull_secret_content), isUpgrade=option.is_upgrade, name=option.membership_ref, connectAgent_namespace=option.namespace, connectAgent_proxy=six.ensure_binary(option.proxy), registry=option.registry, version=option.version)), 'manifest')

    def GenerateConnectAgentManifest(self, option):
        """Generate the YAML manifest to deploy the Connect Agent.

    Args:
      option: an instance of ConnectAgentOption.

    Returns:
      A slice of connect agent manifest resources.
    Raises:
      Error: if the API call to generate connect agent manifest failed.
    """
        client = core_apis.GetClientInstance(API_NAME, self.api_version)
        messages = core_apis.GetMessagesModule(API_NAME, self.api_version)
        encoding.AddCustomJsonFieldMapping(messages.GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest, 'connectAgent_namespace', 'connectAgent.namespace')
        encoding.AddCustomJsonFieldMapping(messages.GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest, 'connectAgent_proxy', 'connectAgent.proxy')
        return self._ManifestResponse(client, messages, option)