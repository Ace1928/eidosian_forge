from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def _AddGenericSectionToSummary(self, location, terms):
    """Helper function for adding sections in the form ['loc1','loc2',...]."""
    section = self.command
    for loc in location:
        section = section.get(loc, {})
        if isinstance(section, str):
            line = section
        elif isinstance(section, list):
            line = ', '.join(sorted(section))
        elif isinstance(section, dict):
            line = ', '.join(sorted(section.keys()))
        else:
            line = six.text_type(section)
    assert line, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
    header = _FormatHeader(location[-1])
    if header:
        self._lines.append(header)
    loc = '.'.join(location)
    self._lines.append(_Snip(line, self.length_per_snippet, terms))