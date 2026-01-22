from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _qformat(self, aline, bline, atags, btags):
    """
        Format "?" output and deal with tabs.

        Example:

        >>> d = Differ()
        >>> results = d._qformat('\\tabcDefghiJkl\\n', '\\tabcdefGhijkl\\n',
        ...                      '  ^ ^  ^      ', '  ^ ^  ^      ')
        >>> for line in results: print(repr(line))
        ...
        '- \\tabcDefghiJkl\\n'
        '? \\t ^ ^  ^\\n'
        '+ \\tabcdefGhijkl\\n'
        '? \\t ^ ^  ^\\n'
        """
    atags = _keep_original_ws(aline, atags).rstrip()
    btags = _keep_original_ws(bline, btags).rstrip()
    yield ('- ' + aline)
    if atags:
        yield f'? {atags}\n'
    yield ('+ ' + bline)
    if btags:
        yield f'? {btags}\n'