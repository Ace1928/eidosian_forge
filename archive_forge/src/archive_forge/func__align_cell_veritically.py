from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _align_cell_veritically(text_lines, num_lines, column_width, row_alignment):
    delta_lines = num_lines - len(text_lines)
    blank = [' ' * column_width]
    if row_alignment == 'bottom':
        return blank * delta_lines + text_lines
    elif row_alignment == 'center':
        top_delta = delta_lines // 2
        bottom_delta = delta_lines - top_delta
        return top_delta * blank + text_lines + bottom_delta * blank
    else:
        return text_lines + blank * delta_lines