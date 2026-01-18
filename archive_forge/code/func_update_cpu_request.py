from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def update_cpu_request(messages, current, value):
    """Configures a cpu request for the current deployment configuration.

  Args:
    messages: the set of proto messages for this feature.
    current: the deployment configuration object being modified.
    value: The value to set the cpu request to. If None, the request will be
      removed. If this is the only request, request requirements will be
      removed. If this is the only requirement, requirements will be removed.

  Returns:
    The modified deployment configuration object.
  """
    if current.containerResources is not None:
        requirements = current.containerResources
    else:
        requirements = messages.PolicyControllerResourceRequirements()
    resource_list = messages.PolicyControllerResourceList()
    if requirements.requests is not None:
        resource_list = requirements.requests
    resource_list.cpu = value
    if resource_list.cpu is None and resource_list.memory is None:
        resource_list = None
    requirements.requests = resource_list
    if requirements.limits is None and requirements.requests is None:
        requirements = None
    current.containerResources = requirements
    return current