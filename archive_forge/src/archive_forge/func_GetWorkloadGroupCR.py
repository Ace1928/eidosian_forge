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
def GetWorkloadGroupCR(self, workload_namespace, workload_name):
    """Get the YAML output of the specified WorkloadGroup CR."""
    if not self._WorkloadGroupCRDExists():
        raise ClusterError('WorkloadGroup CRD is not found in the cluster. Please install Anthos Service Mesh and retry.')
    out, err = self._RunKubectl(['get', 'workloadgroups.networking.istio.io', workload_name, '-n', workload_namespace, '-o', 'yaml'], None)
    if err:
        if 'NotFound' in err:
            raise WorkloadError('WorkloadGroup {} in namespace {} is not found in the cluster. Please create the WorkloadGroup and retry.'.format(workload_name, workload_namespace))
        raise exceptions.Error('Error retrieving WorkloadGroup {} in namespace {}: {}'.format(workload_name, workload_namespace, err))
    return out