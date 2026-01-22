from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
def GenerateInstallManifestForImport(self, location_ref, memberships_id, args):
    """Generates an Attached cluster install manifest for import."""
    req = self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest(parent=location_ref.RelativeName(), attachedClusterId=memberships_id, platformVersion=attached_flags.GetPlatformVersion(args), proxyConfig_kubernetesSecret_name=attached_flags.GetProxySecretName(args), proxyConfig_kubernetesSecret_namespace=attached_flags.GetProxySecretNamespace(args))
    encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_name', 'proxyConfig.kubernetesSecret.name')
    encoding.AddCustomJsonFieldMapping(self._messages.GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest, 'proxyConfig_kubernetesSecret_namespace', 'proxyConfig.kubernetesSecret.namespace')
    return self._service.GenerateAttachedClusterInstallManifest(req)