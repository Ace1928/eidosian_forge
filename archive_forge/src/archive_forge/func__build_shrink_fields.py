import os
import sys
import prettytable
from cliff import utils
from . import base
from cliff import columns
@staticmethod
def _build_shrink_fields(usable_total_width, optimal_width, field_widths, field_names):
    shrink_fields = []
    shrink_remaining = usable_total_width
    for field in field_names:
        w = field_widths[field]
        if w <= optimal_width:
            shrink_remaining -= w
        else:
            shrink_fields.append(field)
    return (shrink_fields, shrink_remaining)