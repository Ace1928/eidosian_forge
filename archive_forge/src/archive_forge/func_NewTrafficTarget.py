from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def NewTrafficTarget(messages, key, percent=None, tag=None):
    """Creates a new TrafficTarget.

  Args:
    messages: The message module that defines TrafficTarget.
    key: The key for the traffic target in the TrafficTargets mapping.
    percent: Optional percent of traffic to assign to the traffic target.
    tag: Optional tag to assign to the traffic target.

  Returns:
    The newly created TrafficTarget.
  """
    if key == LATEST_REVISION_KEY:
        result = messages.TrafficTarget(latestRevision=True, percent=percent, tag=tag)
    else:
        result = messages.TrafficTarget(revisionName=key, percent=percent, tag=tag)
    return result