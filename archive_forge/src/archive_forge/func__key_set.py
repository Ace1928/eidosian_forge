from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
@property
def _key_set(self):
    """A set containing the mapping's keys."""
    return set((GetKey(t) for t in self._m))