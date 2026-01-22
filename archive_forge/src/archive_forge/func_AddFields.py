from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def AddFields(self, fields):
    """Convenience method to add a list of fields."""
    for field in fields:
        self.AddField(field)