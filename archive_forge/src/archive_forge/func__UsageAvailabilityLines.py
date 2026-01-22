from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _UsageAvailabilityLines(actions_grouped_by_kind):
    availability_lines = []
    for action_group in actions_grouped_by_kind:
        if action_group.members:
            availability_line = _CreateAvailabilityLine(header='available {plural}:'.format(plural=action_group.plural), items=action_group.names)
            availability_lines.append(availability_line)
    return availability_lines