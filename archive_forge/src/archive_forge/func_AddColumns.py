from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def AddColumns(self, column_names, kwdss=None):
    """Add a series of columns to this formatter."""
    kwdss = kwdss or [{}] * len(column_names)
    for column_name, kwds in zip(column_names, kwdss):
        self.AddColumn(column_name, **kwds)