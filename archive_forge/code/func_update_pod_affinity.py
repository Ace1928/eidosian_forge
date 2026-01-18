from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def update_pod_affinity(messages, current, value):
    """Configures the pod affinity for the current deployment configuration.

  Args:
    messages: the set of proto messages for this feature.
    current: the deployment configuration object being modified.
    value: The value to set the pod affinity to. If the value is the string
      "none" or value `None`, the pod affinity will be NO_AFFINITY.

  Returns:
    The modified deployment configuration object.
  """
    if value == 'anti':
        current.podAffinity = messages.PolicyControllerPolicyControllerDeploymentConfig.PodAffinityValueValuesEnum.ANTI_AFFINITY
    elif value is None or value == 'none':
        current.podAffinity = messages.PolicyControllerPolicyControllerDeploymentConfig.PodAffinityValueValuesEnum.NO_AFFINITY
    else:
        raise exceptions.Error('invalid pod affinity option {} specified.'.format(value))
    return current