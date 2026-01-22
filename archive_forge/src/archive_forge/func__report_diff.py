from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def _report_diff(self, growth):
    if not growth:
        return '<Unable to calculate growth>'
    lines = []
    width = max((len(name) for name, _, _ in growth))
    for name, count, delta in growth:
        lines.append('%-*s%9d %+9d' % (width, name, count, delta))
    diff = '\n'.join(lines)
    return diff