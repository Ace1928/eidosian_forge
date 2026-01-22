from __future__ import division
import sys
import unicodedata
from functools import reduce
def _splitit(self, line, isheader):
    """Split each element of line to fit the column width

        Each element is turned into a list, result of the wrapping of the
        string to the desired width
        """
    line_wrapped = []
    for cell, width in zip(line, self._width):
        array = []
        for c in cell.split('\n'):
            if c.strip() == '':
                array.append('')
            else:
                array.extend(textwrapper(c, width))
        line_wrapped.append(array)
    max_cell_lines = reduce(max, list(map(len, line_wrapped)))
    for cell, valign in zip(line_wrapped, self._valign):
        if isheader:
            valign = 't'
        if valign == 'm':
            missing = max_cell_lines - len(cell)
            cell[:0] = [''] * int(missing / 2)
            cell.extend([''] * int(missing / 2 + missing % 2))
        elif valign == 'b':
            cell[:0] = [''] * (max_cell_lines - len(cell))
        else:
            cell.extend([''] * (max_cell_lines - len(cell)))
    return line_wrapped