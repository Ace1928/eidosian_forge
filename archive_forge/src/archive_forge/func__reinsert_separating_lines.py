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
def _reinsert_separating_lines(rows, separating_lines):
    if separating_lines:
        for index in separating_lines:
            rows.insert(index, SEPARATING_LINE)