import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def _natural_sort_strings(vals, reverse=False):

    def key(v):
        v_parts = re.split('(\\d+)', v)
        for i in range(len(v_parts)):
            try:
                v_parts[i] = int(v_parts[i])
            except ValueError:
                pass
        return tuple(v_parts)
    return sorted(vals, key=key, reverse=reverse)