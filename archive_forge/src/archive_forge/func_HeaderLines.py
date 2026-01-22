from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def HeaderLines(self):
    """Return an iterator over the row(s) for the column names."""
    aligns = itertools.repeat('c')
    return self.FormatRow(self.column_names, self.header_height, column_alignments=aligns)