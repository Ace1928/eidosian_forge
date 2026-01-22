from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.api_lib.container.fleet.connectgateway import client as gateway_client
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util as hubapi_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util as kconfig
from googlecloudsdk.command_lib.container.fleet import overrides
from googlecloudsdk.command_lib.container.fleet.memberships import errors as memberships_errors
from googlecloudsdk.command_lib.container.fleet.memberships import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def RunGetCredentials(self, membership_id, arg_location, arg_namespace=None):
    container_util.CheckKubectlInstalled()
    project_id = hub_base.HubCommand.Project()
    log.status.Print('Starting to build Gateway kubeconfig...')
    log.status.Print('Current project_id: ' + project_id)
    self.RunIamCheck(project_id, REQUIRED_CLIENT_PERMISSIONS)
    try:
        hub_endpoint_override = properties.VALUES.api_endpoint_overrides.Property('gkehub').Get()
    except properties.NoSuchPropertyError:
        hub_endpoint_override = None
    CheckGatewayApiEnablement(project_id, util.GetConnectGatewayServiceName(hub_endpoint_override, None))
    membership = self.ReadClusterMembership(project_id, arg_location, membership_id)
    collection = 'memberships'
    if project_id == 'gkeconnect-prober':
        pass
    elif hasattr(membership, 'endpoint') and hasattr(membership.endpoint, 'gkeCluster') and membership.endpoint.gkeCluster:
        collection = 'gkeMemberships'
    self.GenerateKubeconfig(util.GetConnectGatewayServiceName(hub_endpoint_override, arg_location), project_id, arg_location, collection, membership_id, arg_namespace)
    msg = 'A new kubeconfig entry "' + self.KubeContext(project_id, arg_location, membership_id, arg_namespace) + '" has been generated and set as the current context.'
    log.status.Print(msg)