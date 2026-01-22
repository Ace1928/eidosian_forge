from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import itertools
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import local
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.core import yaml
import six
def _GenerateKubeConfigs(code_generators):
    """Generate Kubernetes yaml configs.

  Args:
    code_generators: Iterable of KubeConfigGenerator.

  Returns:
    Iterable of dictionaries representing kubernetes yaml configs.
  """
    kube_configs = []
    for code_generator in code_generators:
        kube_configs.extend(code_generator.CreateConfigs())
    deployments = [config for config in kube_configs if config['kind'] == 'Deployment']
    for deployment, code_generator in itertools.product(deployments, code_generators):
        code_generator.ModifyDeployment(deployment)
    for deployment in deployments:
        containers = yaml_helper.GetAll(deployment, ('spec', 'template', 'spec', 'containers'))
        for container, code_generator in itertools.product(containers, code_generators):
            code_generator.ModifyContainer(container)
    return yaml.dump_all(kube_configs)