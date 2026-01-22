import re
import tokenize
from hacking import core
import pycodestyle
def _any_in(line, *sublines):
    for subline in sublines:
        if subline in line:
            return True
    return False