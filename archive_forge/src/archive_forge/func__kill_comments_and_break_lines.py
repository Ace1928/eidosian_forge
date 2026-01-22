from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _kill_comments_and_break_lines(text):
    """Delete []-delimited comments out of a file and break into lines separated by ';' (PRIVATE).

    stripped_text=_kill_comments_and_break_lines(text):
    Nested and multiline comments are allowed. [ and ] symbols within single
    or double quotes are ignored, newline ends a quote, all symbols with quotes are
    treated the same (thus not quoting inside comments like [this character ']' ends a comment])
    Special [&...] and [\\...] comments remain untouched, if not inside standard comment.
    Quotes inside special [& and [\\ are treated as normal characters,
    but no nesting inside these special comments allowed (like [&   [\\   ]]).
    ';' is deleted from end of line.

    NOTE: this function is very slow for large files, and obsolete when using C extension cnexus
    """
    if not text:
        return ''
    contents = iter(text)
    newtext = []
    newline = []
    quotelevel = ''
    speciallevel = False
    commlevel = 0
    t2 = next(contents)
    while True:
        t = t2
        try:
            t2 = next(contents)
        except StopIteration:
            t2 = None
        if t is None:
            break
        if t == quotelevel and (not (commlevel or speciallevel)):
            quotelevel = ''
        elif not quotelevel and (not (commlevel or speciallevel)) and (t == '"' or t == "'"):
            quotelevel = t
        elif not quotelevel and t == '[':
            if t2 in SPECIALCOMMENTS and commlevel == 0 and (not speciallevel):
                speciallevel = True
            else:
                commlevel += 1
        elif not quotelevel and t == ']':
            if speciallevel:
                speciallevel = False
            else:
                commlevel -= 1
                if commlevel < 0:
                    raise NexusError('Nexus formatting error: unmatched ]')
                continue
        if commlevel == 0:
            if t == ';' and (not quotelevel):
                newtext.append(''.join(newline))
                newline = []
            else:
                newline.append(t)
    if newline:
        newtext.append('\n'.join(newline))
    if commlevel > 0:
        raise NexusError('Nexus formatting error: unmatched [')
    return newtext