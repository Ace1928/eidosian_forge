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
def _NameSection(component, info, trace=None, verbose=False):
    """The "Name" section of the help string."""
    current_command = _GetCurrentCommand(trace, include_separators=verbose)
    summary = _GetSummary(info)
    if custom_descriptions.NeedsCustomDescription(component):
        available_space = LINE_LENGTH - SECTION_INDENTATION - len(current_command + ' - ')
        summary = custom_descriptions.GetSummary(component, available_space, LINE_LENGTH)
    if summary:
        text = current_command + ' - ' + summary
    else:
        text = current_command
    return ('NAME', text)