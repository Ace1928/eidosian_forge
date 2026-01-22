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
def VerifyWorkloadSetup(workload_manifest):
    """Verify VM workload setup in the cluster."""
    if not workload_manifest:
        raise WorkloadError('Cannot verify an empty workload from the cluster')
    try:
        workload_data = yaml.load(workload_manifest)
    except yaml.Error as e:
        raise exceptions.Error('Invalid workload from the cluster {}'.format(workload_manifest), e)
    identity_provider_value = _GetNestedKeyFromManifest(workload_data, 'spec', 'metadata', 'annotations', 'security.cloud.google.com/IdentityProvider')
    if identity_provider_value != 'google':
        raise WorkloadError('Unable to find the GCE IdentityProvider in the specified WorkloadGroup. Please make sure the GCE IdentityProvider is specified in the WorkloadGroup.')