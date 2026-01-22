from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import re
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_http as v2_docker_http
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from googlecloudsdk.api_lib.container.images import container_analysis_data_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.core.util import times
import six
from six.moves import map
import six.moves.http_client
def _MakeSummaryRequest(project_id, url_filter):
    """Helper function to make Summary request."""
    client = apis.GetClientInstance('containeranalysis', 'v1alpha1')
    messages = apis.GetMessagesModule('containeranalysis', 'v1alpha1')
    project_ref = resources.REGISTRY.Parse(project_id, collection='cloudresourcemanager.projects')
    req = messages.ContaineranalysisProjectsOccurrencesGetVulnerabilitySummaryRequest(parent=project_ref.RelativeName(), filter=url_filter)
    return client.projects_occurrences.GetVulnerabilitySummary(req)