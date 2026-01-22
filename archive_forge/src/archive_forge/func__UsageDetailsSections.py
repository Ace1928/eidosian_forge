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
def _UsageDetailsSections(component, actions_grouped_by_kind):
    """The usage details sections of the help string."""
    groups, commands, values, indexes = actions_grouped_by_kind
    sections = []
    if groups.members:
        sections.append(_MakeUsageDetailsSection(groups))
    if commands.members:
        sections.append(_MakeUsageDetailsSection(commands))
    if values.members:
        sections.append(_ValuesUsageDetailsSection(component, values))
    if indexes.members:
        sections.append(('INDEXES', _NewChoicesSection('INDEX', indexes.names)))
    return sections