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
def ListCloudRunForAnthosClusters(project):
    """Get all clusters with Cloud Run for Anthos enabled.

  Args:
   project: str optional of project to search for clusters in. Leaving this
     field blank will use the project defined by the corresponding property.

  Returns:
    List of Cluster string names
  """
    crfa_spec = 'projects/%s/locations/global/features/%s' % (project, CLOUDRUN_FEATURE)
    try:
        f = hub_client.HubClient().GetFeature(crfa_spec)
    except api_exceptions.HttpError:
        return []
    cluster_state_obj = _ListAnthosClusterStates(f)
    return [name for name, state in cluster_state_obj.items() if state == 'OK']