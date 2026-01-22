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
def _GetActionsGroupedByKind(component, verbose=False):
    """Gets lists of available actions, grouped by action kind."""
    groups = ActionGroup(name='group', plural='groups')
    commands = ActionGroup(name='command', plural='commands')
    values = ActionGroup(name='value', plural='values')
    indexes = ActionGroup(name='index', plural='indexes')
    members = completion.VisibleMembers(component, verbose=verbose)
    for member_name, member in members:
        member_name = str(member_name)
        if value_types.IsGroup(member):
            groups.Add(name=member_name, member=member)
        if value_types.IsCommand(member):
            commands.Add(name=member_name, member=member)
        if value_types.IsValue(member):
            values.Add(name=member_name, member=member)
    if isinstance(component, (list, tuple)) and component:
        component_len = len(component)
        if component_len < 10:
            indexes.Add(name=', '.join((str(x) for x in range(component_len))))
        else:
            indexes.Add(name='0..{max}'.format(max=component_len - 1))
    return [groups, commands, values, indexes]