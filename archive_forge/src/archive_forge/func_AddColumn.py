from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
def AddColumn(self, column_name, **kwds):
    self._column_names.append(column_name)