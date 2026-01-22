from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def ProjectIdToProjectNumber(project_id):
    """Returns the Cloud project number associated with the `project_id`."""
    crm_message_module = apis.GetMessagesModule('cloudresourcemanager', 'v1')
    resource_manager = apis.GetClientInstance('cloudresourcemanager', 'v1')
    req = crm_message_module.CloudresourcemanagerProjectsGetRequest(projectId=project_id)
    project = resource_manager.projects.Get(req)
    return project.projectNumber