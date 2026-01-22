from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseEnvironment(environment_name):
    """Parse out an environment resource using configuration properties.

  Args:
    environment_name: str, the environment's ID, absolute name, or relative name
  Returns:
    Environment: the parsed environment resource
  """
    return resources.REGISTRY.Parse(environment_name, params={'projectsId': GetProject, 'locationsId': GetLocation}, collection='composer.projects.locations.environments')