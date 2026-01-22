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
def UsageText(component, trace=None, verbose=False):
    """Returns usage text for the given component.

  Args:
    component: The component to determine the usage text for.
    trace: The Fire trace object containing all metadata of current execution.
    verbose: Whether to display the usage text in verbose mode.

  Returns:
    String suitable for display in an error screen.
  """
    output_template = 'Usage: {continued_command}\n{availability_lines}\nFor detailed information on this command, run:\n  {help_command}'
    if trace:
        command = trace.GetCommand()
        needs_separating_hyphen_hyphen = trace.NeedsSeparatingHyphenHyphen()
    else:
        command = None
        needs_separating_hyphen_hyphen = False
    if not command:
        command = ''
    continued_command = command
    spec = inspectutils.GetFullArgSpec(component)
    metadata = decorators.GetMetadata(component)
    actions_grouped_by_kind = _GetActionsGroupedByKind(component, verbose=verbose)
    possible_actions = _GetPossibleActions(actions_grouped_by_kind)
    continuations = []
    if possible_actions:
        continuations.append(_GetPossibleActionsUsageString(possible_actions))
    availability_lines = _UsageAvailabilityLines(actions_grouped_by_kind)
    if callable(component):
        callable_items = _GetCallableUsageItems(spec, metadata)
        if callable_items:
            continuations.append(' '.join(callable_items))
        elif trace:
            continuations.append(trace.separator)
        availability_lines.extend(_GetCallableAvailabilityLines(spec))
    if continuations:
        continued_command += ' ' + ' | '.join(continuations)
    help_command = command + (' -- ' if needs_separating_hyphen_hyphen else ' ') + '--help'
    return output_template.format(continued_command=continued_command, availability_lines=''.join(availability_lines), help_command=help_command)