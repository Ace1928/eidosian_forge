from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _ValidateCurrentTraffic(self, existing_percent_targets):
    """Validate current traffic targets."""
    percent = 0
    for target in existing_percent_targets:
        percent += target.percent
    if percent != 100:
        raise ValueError('Current traffic allocation of %s is not 100 percent' % percent)
    for target in existing_percent_targets:
        if target.percent < 0:
            raise ValueError('Current traffic for target %s is negative (%s)' % (GetKey(target), target.percent))