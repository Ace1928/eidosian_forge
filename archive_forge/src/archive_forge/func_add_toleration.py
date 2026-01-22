from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def add_toleration(messages, current, key_value, effect):
    """Adds a toleration to the current deployment configuration.

  Args:
    messages: the set of proto messages for this feature.
    current: the deployment configuration object being modified.
    key_value: the key-and-optional-value string specifying the toleration key
      and value.
    effect: Optional. If included, will set the effect value on the toleration.

  Returns:
    The modified deployment configuration object.
  """
    toleration = messages.PolicyControllerToleration()
    key, value, operator = _parse_key_value(key_value)
    toleration.operator = operator
    toleration.key = key
    if value is not None:
        toleration.value = value
    if effect is not None:
        toleration.effect = effect
    tolerations = []
    if current.podTolerations is not None:
        tolerations = current.podTolerations
    tolerations.append(toleration)
    current.podTolerations = tolerations
    return current