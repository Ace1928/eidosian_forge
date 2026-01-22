from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _OtherTargets(self, key):
    """Gets all targets that do not match the given key."""
    return [t for t in self._m if GetKey(t) != key]