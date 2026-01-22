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
def _MakeUsageDetailsSection(action_group):
    """Creates a usage details section for the provided action group."""
    item_strings = []
    for name, member in action_group.GetItems():
        info = inspectutils.Info(member)
        item = name
        docstring_info = info.get('docstring_info')
        if docstring_info and (not custom_descriptions.NeedsCustomDescription(member)):
            summary = docstring_info.summary
        elif custom_descriptions.NeedsCustomDescription(member):
            summary = custom_descriptions.GetSummary(member, LINE_LENGTH - SECTION_INDENTATION, LINE_LENGTH)
        else:
            summary = None
        item = _CreateItem(name, summary)
        item_strings.append(item)
    return (action_group.plural.upper(), _NewChoicesSection(action_group.name.upper(), item_strings))