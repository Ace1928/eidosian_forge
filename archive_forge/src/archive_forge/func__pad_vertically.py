import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _pad_vertically(lines, max_lines):
    pad_line = [' ' * len(lines[0])]
    return lines + pad_line * (max_lines - len(lines))