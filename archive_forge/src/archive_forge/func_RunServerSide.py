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
def RunServerSide(self, membership_id: str, arg_location: str, force_use_agent: bool=False):
    """RunServerSide generates credentials using server-side kubeconfig generation.

    Args:
      membership_id: The short name of the membership to generate credentials
        for.
      arg_location: The location of the membership to generate credentials for.
      force_use_agent: Whether to force the use of Connect Agent in generated
        credentials.
    """
    log.status.Print('Fetching Gateway kubeconfig...')
    container_util.CheckKubectlInstalled()
    project_id = hub_base.HubCommand.Project()
    project_number = hub_base.HubCommand.Project(number=True)
    self.RunIamCheck(project_id, REQUIRED_SERVER_PERMISSIONS)
    with overrides.RegionalGatewayEndpoint(arg_location):
        client = gateway_client.GatewayClient(self.ReleaseTrack())
        resp = client.GenerateCredentials(name=f'projects/{project_number}/locations/{arg_location}/memberships/{membership_id}', force_use_agent=force_use_agent)
    new = kconfig.Kubeconfig.LoadFromBytes(resp.kubeconfig)
    kubeconfig = kconfig.Kubeconfig.Default()
    kubeconfig.Merge(new, overwrite=True)
    kubeconfig.SetCurrentContext(list(new.contexts.keys())[0])
    kubeconfig.SaveToFile()
    msg = f'A new kubeconfig entry "{kubeconfig.current_context}" has been generated and set as the current context.'
    log.status.Print(msg)