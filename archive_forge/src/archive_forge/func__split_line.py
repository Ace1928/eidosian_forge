from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _split_line(self, data_list, line_num, text):
    """Builds list of text lines by splitting text lines at wrap point

        This function will determine if the input text line needs to be
        wrapped (split) into separate lines.  If so, the first wrap point
        will be determined and the first line appended to the output
        text line list.  This function is used recursively to handle
        the second part of the split line to further split it.
        """
    if not line_num:
        data_list.append((line_num, text))
        return
    size = len(text)
    max = self._wrapcolumn
    if size <= max or size - text.count('\x00') * 3 <= max:
        data_list.append((line_num, text))
        return
    i = 0
    n = 0
    mark = ''
    while n < max and i < size:
        if text[i] == '\x00':
            i += 1
            mark = text[i]
            i += 1
        elif text[i] == '\x01':
            i += 1
            mark = ''
        else:
            i += 1
            n += 1
    line1 = text[:i]
    line2 = text[i:]
    if mark:
        line1 = line1 + '\x01'
        line2 = '\x00' + mark + line2
    data_list.append((line_num, line1))
    self._split_line(data_list, '>', line2)