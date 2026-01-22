from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.container.fleet import util as hub_util
from googlecloudsdk.api_lib.resourcesettings import service as resourcesettings_service
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.runtime_config import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetClusterRef(cluster, project=None):
    """Returns a ref for the specified cluster.

  Args:
    cluster: container_CONTAINER_API_VERSION_messages.Cluster object
    project: str optional project which overrides the default

  Returns:
    A Resource object
  """
    if not project:
        project = properties.VALUES.core.project.Get(required=True)
    return resources.REGISTRY.Parse(cluster.name, params={'projectId': project, 'zone': cluster.zone}, collection='container.projects.zones.clusters')