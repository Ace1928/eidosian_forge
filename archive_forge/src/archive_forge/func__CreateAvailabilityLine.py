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
def _CreateAvailabilityLine(header, items, header_indent=2, items_indent=25, line_length=LINE_LENGTH):
    items_width = line_length - items_indent
    items_text = '\n'.join(formatting.WrappedJoin(items, width=items_width))
    indented_items_text = formatting.Indent(items_text, spaces=items_indent)
    indented_header = formatting.Indent(header, spaces=header_indent)
    return indented_header + indented_items_text[len(indented_header):] + '\n'