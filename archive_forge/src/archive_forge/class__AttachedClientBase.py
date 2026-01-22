from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
class _AttachedClientBase(client.ClientBase):
    """Base class for Attached gkemulticloud API clients."""

    def _Cluster(self, cluster_ref, args):
        cluster_type = self._messages.GoogleCloudGkemulticloudV1AttachedCluster
        kwargs = {'annotations': self._Annotations(args, cluster_type), 'binaryAuthorization': self._BinaryAuthorization(args), 'platformVersion': attached_flags.GetPlatformVersion(args), 'fleet': self._Fleet(args), 'name': cluster_ref.attachedClustersId, 'description': flags.GetDescription(args), 'oidcConfig': self._OidcConfig(args), 'distribution': attached_flags.GetDistribution(args), 'authorization': self._Authorization(args), 'loggingConfig': flags.GetLogging(args, True), 'monitoringConfig': flags.GetMonitoringConfig(args), 'proxyConfig': self._ProxyConfig(args)}
        return self._messages.GoogleCloudGkemulticloudV1AttachedCluster(**kwargs) if any(kwargs.values()) else None

    def _OidcConfig(self, args):
        kwargs = {'issuerUrl': attached_flags.GetIssuerUrl(args)}
        oidc = attached_flags.GetOidcJwks(args)
        if oidc:
            kwargs['jwks'] = oidc.encode(encoding='utf-8')
        return self._messages.GoogleCloudGkemulticloudV1AttachedOidcConfig(**kwargs) if any(kwargs.values()) else None

    def _ProxyConfig(self, args):
        secret_name = attached_flags.GetProxySecretName(args)
        secret_namespace = attached_flags.GetProxySecretNamespace(args)
        if secret_name or secret_namespace:
            kwargs = {'kubernetesSecret': self._messages.GoogleCloudGkemulticloudV1KubernetesSecret(name=secret_name, namespace=secret_namespace)}
            return self._messages.GoogleCloudGkemulticloudV1AttachedProxyConfig(**kwargs)
        return None

    def _Authorization(self, args):
        admin_users = attached_flags.GetAdminUsers(args)
        admin_groups = flags.GetAdminGroups(args)
        if not admin_users and (not admin_groups):
            return None
        kwargs = {}
        if admin_users:
            kwargs['adminUsers'] = [self._messages.GoogleCloudGkemulticloudV1AttachedClusterUser(username=u) for u in admin_users]
        if admin_groups:
            kwargs['adminGroups'] = [self._messages.GoogleCloudGkemulticloudV1AttachedClusterGroup(group=g) for g in admin_groups]
        if not any(kwargs.values()):
            return None
        return self._messages.GoogleCloudGkemulticloudV1AttachedClustersAuthorization(**kwargs)