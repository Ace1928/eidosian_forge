from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def update_mem_limit(messages, current, value):
    """Configures a memory limit for the current deployment configuration.

  Args:
    messages: the set of proto messages for this feature.
    current: the deployment configuration object being modified.
    value: The value to set the memory limit to. If None, the limit will be
      removed. If this is the only limit, limit requirements will be removed. If
      this is the only requirement, requirements will be removed.

  Returns:
    The modified deployment configuration object.
  """
    if current.containerResources is not None:
        requirements = current.containerResources
    else:
        requirements = messages.PolicyControllerResourceRequirements()
    resource_list = messages.PolicyControllerResourceList()
    if requirements.limits is not None:
        resource_list = requirements.limits
    resource_list.memory = value
    if resource_list.cpu is None and resource_list.memory is None:
        resource_list = None
    requirements.limits = resource_list
    if requirements.limits is None and requirements.requests is None:
        requirements = None
    current.containerResources = requirements
    return current