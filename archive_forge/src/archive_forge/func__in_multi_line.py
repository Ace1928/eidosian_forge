import re
from mako import exceptions
def _in_multi_line(self, line):
    """return true if the given line is part of a multi-line block,
        via backslash or triple-quote."""
    current_state = self.backslashed or self.triplequoted
    self.backslashed = bool(re.search('\\\\$', line))
    triples = len(re.findall('\\"\\"\\"|\\\'\\\'\\\'', line))
    if triples == 1 or triples % 2 != 0:
        self.triplequoted = not self.triplequoted
    return current_state