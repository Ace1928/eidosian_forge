from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
class LocationsClient(client.ClientBase):
    """Client for managing locations."""

    def __init__(self, **kwargs):
        super(LocationsClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations

    def GetAwsServerConfig(self, location_ref):
        """Gets server config for Anthos on AWS."""
        req = self._messages.GkemulticloudProjectsLocationsGetAwsServerConfigRequest(name=location_ref.RelativeName() + '/awsServerConfig')
        return self._service.GetAwsServerConfig(req)

    def GetAzureServerConfig(self, location_ref):
        """Gets server config for Anthos on Azure."""
        req = self._messages.GkemulticloudProjectsLocationsGetAzureServerConfigRequest(name=location_ref.RelativeName() + '/azureServerConfig')
        return self._service.GetAzureServerConfig(req)

    def GetAttachedServerConfig(self, location_ref):
        """Gets server config for Anthos Attached Clusters."""
        req = self._messages.GkemulticloudProjectsLocationsGetAttachedServerConfigRequest(name=location_ref.RelativeName() + '/attachedServerConfig')
        return self._service.GetAttachedServerConfig(req)

    def GenerateInstallManifest(self, cluster_ref, args):
        """Generates an Attached cluster install manifest."""
        req = self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest(parent=cluster_ref.Parent().RelativeName(), attachedClusterId=cluster_ref.attachedClustersId, platformVersion=attached_flags.GetPlatformVersion(args), proxyConfig_kubernetesSecret_name=attached_flags.GetProxySecretName(args), proxyConfig_kubernetesSecret_namespace=attached_flags.GetProxySecretNamespace(args))
        encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_name', 'proxyConfig.kubernetesSecret.name')
        encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_namespace', 'proxyConfig.kubernetesSecret.namespace')
        return self._service.GenerateAttachedClusterInstallManifest(req)

    def GenerateInstallManifestForImport(self, location_ref, memberships_id, args):
        """Generates an Attached cluster install manifest for import."""
        req = self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest(parent=location_ref.RelativeName(), attachedClusterId=memberships_id, platformVersion=attached_flags.GetPlatformVersion(args), proxyConfig_kubernetesSecret_name=attached_flags.GetProxySecretName(args), proxyConfig_kubernetesSecret_namespace=attached_flags.GetProxySecretNamespace(args))
        encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_name', 'proxyConfig.kubernetesSecret.name')
        encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_namespace', 'proxyConfig.kubernetesSecret.namespace')
        return self._service.GenerateAttachedClusterInstallManifest(req)