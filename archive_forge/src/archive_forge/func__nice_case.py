import re
from Bio import File
def _nice_case(line):
    """Make A Lowercase String With Capitals (PRIVATE)."""
    line_lower = line.lower()
    s = ''
    i = 0
    nextCap = 1
    while i < len(line_lower):
        c = line_lower[i]
        if c >= 'a' and c <= 'z' and nextCap:
            c = c.upper()
            nextCap = 0
        elif c in ' .,;:\t-_':
            nextCap = 1
        s += c
        i += 1
    return s