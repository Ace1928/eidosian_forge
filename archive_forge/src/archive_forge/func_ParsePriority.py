from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
import six
def ParsePriority(priority):
    """Converts a priority to an integer."""
    if priority == 'default':
        priority = DEFAULT_RULE_PRIORITY
    try:
        priority_int = int(priority)
        if priority_int <= 0 or priority_int > DEFAULT_RULE_PRIORITY:
            raise exceptions.InvalidArgumentException('priority', 'Priority must be between 1 and {0} inclusive.'.format(DEFAULT_RULE_PRIORITY))
        return priority_int
    except ValueError:
        raise exceptions.InvalidArgumentException('priority', 'Priority should be an integer value or `default`.')