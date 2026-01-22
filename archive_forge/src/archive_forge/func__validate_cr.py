from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _validate_cr(yaml_string):
    """Validate the parsed cloudrun YAML.

  Args:
    yaml_string: The YAML string to validate.
  """
    try:
        cloudrun_cr = yaml.load(yaml_string)
    except yaml.Error as e:
        raise exceptions.Error('Invalid cloudrun yaml {}'.format(yaml_string), e)
    if not isinstance(cloudrun_cr, dict):
        raise exceptions.Error('Invalid CloudRun template.')
    if 'apiVersion' not in cloudrun_cr:
        raise exceptions.Error('The resource is missing a required field "apiVersion".')
    if cloudrun_cr['apiVersion'] != 'operator.run.cloud.google.com/v1alpha1':
        raise exceptions.Error('The resource "apiVersion" field must be set to: "operator.run.cloud.google.com/v1alpha1". If you believe the apiVersion is correct, you may need to upgrade your gcloud installation.')
    if 'kind' not in cloudrun_cr:
        raise exceptions.Error('The resource is missing a required field "kind".')
    if cloudrun_cr['kind'] != 'CloudRun':
        raise exceptions.Error('The resource "kind" field must be set to: "CloudRun".')
    if 'metadata' not in cloudrun_cr:
        raise cloudrun_cr.Error('The resource is missing a required field "metadata".')
    metadata = cloudrun_cr['metadata']
    if 'name' not in metadata or metadata['name'] != 'cloud-run':
        raise exceptions.Error('The resource "metadata.name" field must be set to "cloud-run"')