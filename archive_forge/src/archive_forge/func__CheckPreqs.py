from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
from googlecloudsdk.api_lib.container import kubeconfig as kubeconfig_util
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.container.fleet import gateway
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
def _CheckPreqs(private_endpoint=False):
    """Checks the prerequisites to run get-credentials commands."""
    util.CheckKubectlInstalled()
    if not private_endpoint:
        project_id = properties.VALUES.core.project.GetOrFail()
        gateway.CheckGatewayApiEnablement(project_id, _GetConnectGatewayEndpoint())