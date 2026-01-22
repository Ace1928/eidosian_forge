from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetTraceSinkResource(sink_name, project):
    """Returns the appropriate sink resource based on args."""
    return resources.REGISTRY.Parse(sink_name, params={'projectsId': GetProjectNumber(project)}, collection='cloudtrace.projects.traceSinks')