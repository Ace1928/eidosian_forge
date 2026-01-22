from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _tab_newline_replace(self, fromlines, tolines):
    """Returns from/to line lists with tabs expanded and newlines removed.

        Instead of tab characters being replaced by the number of spaces
        needed to fill in to the next tab stop, this function will fill
        the space with tab characters.  This is done so that the difference
        algorithms can identify changes in a file when tabs are replaced by
        spaces and vice versa.  At the end of the HTML generation, the tab
        characters will be replaced with a nonbreakable space.
        """

    def expand_tabs(line):
        line = line.replace(' ', '\x00')
        line = line.expandtabs(self._tabsize)
        line = line.replace(' ', '\t')
        return line.replace('\x00', ' ').rstrip('\n')
    fromlines = [expand_tabs(line) for line in fromlines]
    tolines = [expand_tabs(line) for line in tolines]
    return (fromlines, tolines)