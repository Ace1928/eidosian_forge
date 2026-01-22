import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def _find_optimal(rlist, row_first: bool, separator_size: int, displaywidth: int):
    """Calculate optimal info to columnize a list of string"""
    for max_rows in range(1, len(rlist) + 1):
        col_widths = list(map(max, _col_chunks(rlist, max_rows, row_first)))
        sumlength = sum(col_widths)
        ncols = len(col_widths)
        if sumlength + separator_size * (ncols - 1) <= displaywidth:
            break
    return {'num_columns': ncols, 'optimal_separator_width': (displaywidth - sumlength) // (ncols - 1) if ncols - 1 else 0, 'max_rows': max_rows, 'column_widths': col_widths}