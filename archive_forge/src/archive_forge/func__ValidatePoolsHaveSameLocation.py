from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _ValidatePoolsHaveSameLocation(pools):
    """Validates that all pools specify an identical location."""
    if not pools:
        return
    initial_locations = None
    for pool in pools:
        if pool.nodePoolConfig is not None:
            locations = pool.nodePoolConfig.locations
            if initial_locations is None:
                initial_locations = locations
                continue
            elif initial_locations != locations:
                raise exceptions.InvalidArgumentException('--pools', 'All pools must have identical locations.')