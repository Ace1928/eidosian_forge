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
def RetrieveMeshConfig(self, revision):
    """Retrieves the MeshConfig for the ASM revision."""
    if revision == 'default':
        mesh_config_name = 'istio'
    else:
        mesh_config_name = 'istio-{}'.format(revision)
    out, err = self._RunKubectl(['get', 'configmap', mesh_config_name, '-n', 'istio-system', '-o', 'jsonpath={.data.mesh}'], None)
    if err:
        if 'NotFound' in err:
            raise ClusterError('Anthos Service Mesh revision {} is not found in the cluster. Please install Anthos Service Mesh and try again.'.format(revision))
        raise exceptions.Error('Error retrieving the mesh config from the cluster: {}'.format(err))
    try:
        mesh_config = yaml.load(out)
    except yaml.Error:
        raise exceptions.Error('Invalid mesh config from the cluster: {}'.format(out))
    return mesh_config