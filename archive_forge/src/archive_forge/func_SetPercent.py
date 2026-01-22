from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def SetPercent(self, key, percent):
    """Set the given percent in the traffic targets.

    Moves any tags on existing targets with the specified key to zero percent
    targets.

    Args:
      key: Name of the revision (or "LATEST") to set the percent for.
      percent: Percent of traffic to set.
    """
    existing = self.get(key)
    if existing:
        new_targets = [NewTrafficTarget(self._messages, key, tag=t.tag) for t in existing if t.tag]
        new_targets.append(NewTrafficTarget(self._messages, key, percent))
        self[key] = new_targets
    else:
        self._m.append(NewTrafficTarget(self._messages, key, percent))