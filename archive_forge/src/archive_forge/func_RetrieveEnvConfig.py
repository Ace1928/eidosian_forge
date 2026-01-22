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
def RetrieveEnvConfig(self, revision):
    """Retrieves the env-* config map for MCP."""
    if revision == 'default':
        env_config_name = 'env'
    else:
        env_config_name = 'env-{}'.format(revision)
    out, err = self._RunKubectl(['get', 'configmap', env_config_name, '-n', 'istio-system', '-o', 'jsonpath={.data}'], None)
    if err:
        if 'NotFound' in err:
            raise ClusterError('Managed Control Plane revision {} is not found in the cluster. Please install Managed Control Plane and try again.'.format(revision))
        raise exceptions.Error('Error retrieving the config map {} from the cluster: {}'.format(env_config_name, err))
    try:
        env_config = yaml.load(out)
    except yaml.Error:
        raise exceptions.Error('Invalid config map from the cluster: {}'.format(out))
    return env_config