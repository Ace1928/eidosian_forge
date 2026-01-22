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
def _GetPossibleActions(actions_grouped_by_kind):
    """The list of possible action kinds."""
    possible_actions = []
    for action_group in actions_grouped_by_kind:
        if action_group.members:
            possible_actions.append(action_group.name)
    return possible_actions