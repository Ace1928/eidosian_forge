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
def _DescriptionSection(component, info):
    """The "Description" sections of the help string.

  Args:
    component: The component to produce the description section for.
    info: The info dict for the component of interest.

  Returns:
    Returns the description if available. If not, returns the summary.
    If neither are available, returns None.
  """
    if custom_descriptions.NeedsCustomDescription(component):
        available_space = LINE_LENGTH - SECTION_INDENTATION
        description = custom_descriptions.GetDescription(component, available_space, LINE_LENGTH)
        summary = custom_descriptions.GetSummary(component, available_space, LINE_LENGTH)
    else:
        description = _GetDescription(info)
        summary = _GetSummary(info)
    text = description or summary or None
    if text:
        return ('DESCRIPTION', text)
    else:
        return None