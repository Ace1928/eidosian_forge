from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class AdminClustersClient(_BareMetalAdminClusterClient):
    """Client for admin clusters in gkeonprem bare metal API."""

    def __init__(self, **kwargs):
        super(AdminClustersClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_bareMetalAdminClusters

    def Enroll(self, args: parser_extensions.Namespace):
        """Enrolls an admin cluster to Anthos on bare metal."""
        kwargs = {'membership': self._admin_cluster_membership_name(args), 'bareMetalAdminClusterId': self._admin_cluster_id(args)}
        req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersEnrollRequest(parent=self._admin_cluster_parent(args), enrollBareMetalAdminClusterRequest=messages.EnrollBareMetalAdminClusterRequest(**kwargs))
        return self._service.Enroll(req)

    def Unenroll(self, args: parser_extensions.Namespace):
        """Unenrolls an Anthos on bare metal admin cluster."""
        kwargs = {'name': self._admin_cluster_name(args), 'allowMissing': self.GetFlag(args, 'allow_missing'), 'validateOnly': self.GetFlag(args, 'validate_only'), 'ignoreErrors': self.GetFlag(args, 'ignore_errors')}
        req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersUnenrollRequest(**kwargs)
        return self._service.Unenroll(req)

    def List(self, args: parser_extensions.Namespace):
        """Lists admin clusters in the GKE On-Prem bare metal API."""
        project = args.project if args.project else properties.VALUES.core.project.Get()
        parent = 'projects/{project}/locations/{location}'.format(project=project, location='us-west1')
        dummy_request = messages.GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest(parent=parent)
        _ = self._service.QueryVersionConfig(dummy_request)
        if 'location' not in args.GetSpecifiedArgsDict() and (not properties.VALUES.container_bare_metal.location.Get()):
            args.location = '-'
        list_req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersListRequest(parent=self._location_name(args))
        return list_pager.YieldFromList(self._service, list_req, field='bareMetalAdminClusters', batch_size=getattr(args, 'page_size', 100), limit=getattr(args, 'limit', None), batch_size_attribute='pageSize')

    def QueryVersionConfig(self, args: parser_extensions.Namespace):
        """Query Anthos on bare metal admin version configuration."""
        kwargs = {'upgradeConfig_clusterName': self._admin_cluster_name(args), 'parent': self._location_ref(args).RelativeName()}
        encoding.AddCustomJsonFieldMapping(messages.GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest, 'upgradeConfig_clusterName', 'upgradeConfig.clusterName')
        req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest(**kwargs)
        return self._service.QueryVersionConfig(req)

    def Create(self, args: parser_extensions.Namespace):
        """Creates an admin cluster in Anthos on bare metal."""
        kwargs = {'parent': self._admin_cluster_parent(args), 'validateOnly': getattr(args, 'validate_only', False), 'bareMetalAdminCluster': self._bare_metal_admin_cluster(args), 'bareMetalAdminClusterId': self._admin_cluster_id(args)}
        req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersCreateRequest(**kwargs)
        return self._service.Create(req)

    def Update(self, args: parser_extensions.Namespace):
        """Updates an admin cluster in Anthos on bare metal."""
        kwargs = {'name': self._admin_cluster_name(args), 'updateMask': update_mask.get_update_mask(args, update_mask.BARE_METAL_ADMIN_CLUSTER_ARGS_TO_UPDATE_MASKS), 'validateOnly': getattr(args, 'validate_only', False), 'bareMetalAdminCluster': self._bare_metal_admin_cluster_for_update(args)}
        req = messages.GkeonpremProjectsLocationsBareMetalAdminClustersPatchRequest(**kwargs)
        return self._service.Patch(req)

    def _bare_metal_admin_cluster_for_update(self, args: parser_extensions.Namespace):
        """Constructs proto message BareMetalAdminCluster."""
        kwargs = {'description': getattr(args, 'description', None), 'bareMetalVersion': getattr(args, 'version', None), 'networkConfig': self._network_config(args), 'controlPlane': self._control_plane_config(args), 'loadBalancer': self._load_balancer_config(args), 'storage': self._storage_config(args), 'proxy': self._proxy_config(args), 'clusterOperations': self._cluster_operations_config(args), 'maintenanceConfig': self._maintenance_config(args), 'nodeConfig': self._workload_node_config(args), 'nodeAccessConfig': self._node_access_config(args), 'binaryAuthorization': self._binary_authorization(args)}
        if any(kwargs.values()):
            return messages.BareMetalAdminCluster(**kwargs)
        return None