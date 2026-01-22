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
def _SynopsisSection(component, actions_grouped_by_kind, spec, metadata, trace=None):
    """The "Synopsis" section of the help string."""
    current_command = _GetCurrentCommand(trace=trace, include_separators=True)
    possible_actions = _GetPossibleActions(actions_grouped_by_kind)
    continuations = []
    if possible_actions:
        continuations.append(_GetPossibleActionsString(possible_actions))
    if callable(component):
        callable_continuation = _GetArgsAndFlagsString(spec, metadata)
        if callable_continuation:
            continuations.append(callable_continuation)
        elif trace:
            continuations.append(trace.separator)
    continuation = ' | '.join(continuations)
    synopsis_template = '{current_command} {continuation}'
    text = synopsis_template.format(current_command=current_command, continuation=continuation)
    return ('SYNOPSIS', text)