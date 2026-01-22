from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _RetrieveDiscoveryAddress(mesh_config):
    """Get the discovery address used in the MCP installation.

  Args:
    mesh_config: A mesh config from the cluster.

  Returns:
    The discovery address.
  """
    proxy_config = _RetrieveProxyConfig(is_mcp=True, mesh_config=mesh_config)
    if proxy_config is None:
        proxy_config = {}
    return proxy_config.get('discoveryAddress', _MCP_ADDRESS)