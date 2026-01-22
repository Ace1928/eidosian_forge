from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ValidateConfig(manifests):
    """Validates the manifests.

  Args:
     manifests: [str], the list of parsed resource yaml definitions.

  Raises:
    exceptions.CloudDeployConfigError, if there are errors in the manifests
    (e.g. required field is missing, duplicate resource names).
  """
    resource_type_to_names = collections.defaultdict(list)
    for manifest in manifests:
        api_version = manifest.get('apiVersion')
        if not api_version:
            raise exceptions.CloudDeployConfigError('missing required field .apiVersion')
        resource_type = manifest.get('kind')
        if resource_type is None:
            raise exceptions.CloudDeployConfigError('missing required field .kind')
        api_version = manifest['apiVersion']
        if api_version not in {API_VERSION_V1BETA1, API_VERSION_V1}:
            raise exceptions.CloudDeployConfigError('api version {} not supported'.format(api_version))
        metadata = manifest.get('metadata')
        if not metadata or not metadata.get(NAME_FIELD):
            raise exceptions.CloudDeployConfigError('missing required field .metadata.name in {}'.format(manifest.get('kind')))
        resource_type_to_names[resource_type].append(metadata.get(NAME_FIELD))
    _CheckDuplicateResourceNames(resource_type_to_names)