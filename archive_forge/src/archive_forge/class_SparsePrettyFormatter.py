from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
class SparsePrettyFormatter(PrettyFormatter):
    """Formats output as a table with a header and separator line."""

    def __init__(self, **kwds):
        """Initialize a new SparsePrettyFormatter."""
        default_kwds = {'junction_char': ' ', 'vertical_char': ' '}
        default_kwds.update(kwds)
        super(SparsePrettyFormatter, self).__init__(**default_kwds)

    def __unicode__(self):
        if self or not self.skip_header_when_empty:
            lines = itertools.chain(self.FormatHeader(), self.FormatRows())
        else:
            lines = []
        return '\n'.join(lines)

    def FormatHeader(self):
        """Return an iterator over the header lines for this table."""
        return itertools.chain(self.HeaderLines(), self.FormatHrule())