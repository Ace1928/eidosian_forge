from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
def GenerateInstallManifest(self, cluster_ref, args):
    """Generates an Attached cluster install manifest."""
    req = self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest(parent=cluster_ref.Parent().RelativeName(), attachedClusterId=cluster_ref.attachedClustersId, platformVersion=attached_flags.GetPlatformVersion(args), proxyConfig_kubernetesSecret_name=attached_flags.GetProxySecretName(args), proxyConfig_kubernetesSecret_namespace=attached_flags.GetProxySecretNamespace(args))
    encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_name', 'proxyConfig.kubernetesSecret.name')
    encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_namespace', 'proxyConfig.kubernetesSecret.namespace')
    return self._service.GenerateAttachedClusterInstallManifest(req)