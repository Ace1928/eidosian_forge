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
def RetrieveWorkloadRevision(namespace_manifest):
    """Retrieve the Anthos Service Mesh revision for the workload."""
    if not namespace_manifest:
        raise WorkloadError('Cannot verify an empty namespace from the cluster')
    try:
        namespace_data = yaml.load(namespace_manifest)
    except yaml.Error as e:
        raise exceptions.Error('Invalid namespace from the cluster {}'.format(namespace_manifest), e)
    workload_revision = _GetNestedKeyFromManifest(namespace_data, 'metadata', 'labels', 'istio.io/rev')
    if not workload_revision:
        raise WorkloadError('Workload namespace does not have an Anthos Service Mesh revision label. Please make sure the namespace is labeled and try again.')
    return workload_revision