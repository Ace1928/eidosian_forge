from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _ValidateUniqueNames(pools):
    """Validates that pools have unique names."""
    used_names = set()
    for pool in pools:
        name = pool.nodePool
        if name in used_names:
            raise exceptions.InvalidArgumentException('--pools', 'Pool name "%s" used more than once.' % name)
        used_names.add(name)