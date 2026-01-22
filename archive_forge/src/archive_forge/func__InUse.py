from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def _InUse(resource):
    """Set of all secret names (local names & remote aliases) in use.

  Args:
    resource: ContainerResource

  Returns:
    List of local names and remote aliases.
  """
    env_vars = itertools.chain.from_iterable((container.env_vars.secrets.values() for container in resource.template.containers.values()))
    return frozenset(itertools.chain((source.secretName for source in resource.template.volumes.secrets.values()), (source.secretKeyRef.name for source in env_vars)))