from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def _ClosureFor(self, ids, adjacencies, component_filter):
    """Calculates a connected closure for the components with the given ids.

    Performs a breadth first search starting with the given component ids, and
    returns the set of components reachable via the given adjacency map.

    Args:
      ids: [str], The component ids to get the closure for.
      adjacencies: {str: set}, Map of component ids to the set of their
        adjacent component ids.
      component_filter: schemas.Component -> bool, A function applied to
        components that determines whether or not to include them in the
        closure.

    Returns:
      set of str, The set of component ids in the closure.
    """
    closure = set()
    to_process = collections.deque(ids)
    while to_process:
        current = to_process.popleft()
        if current not in self.components or current in closure:
            continue
        if not component_filter(self.components[current]):
            continue
        closure.add(current)
        to_process.extend(adjacencies[current])
    return closure