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
def MultiTenantClustersForProject(project_id, cluster_location):
    """Returns a list of clusters accounting for multi-tenant projects.

  This function can also be used for non-multitenant projects and will
  operate on the single passed-in project_id.

  Args:
    project_id: The id of the project, which may or may not be multi-tenant
    cluster_location: The zone or region of the cluster

  Returns:
    A list of cluster refs
  """
    project_ids = _MultiTenantProjectsIfEnabled(project_id)
    project_ids.insert(0, project_id)
    return _ClustersForProjectIds(project_ids, cluster_location)