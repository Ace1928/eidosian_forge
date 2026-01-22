import re
import tokenize
from hacking import core
import pycodestyle
def _starts_with_any(line, *prefixes):
    for prefix in prefixes:
        if line.startswith(prefix):
            return True
    return False