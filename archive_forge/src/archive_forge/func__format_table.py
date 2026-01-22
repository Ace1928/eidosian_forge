from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _format_table(fmt, headers, rows, colwidths, colaligns, is_multiline, rowaligns):
    """Produce a plain-text representation of the table."""
    lines = []
    hidden = fmt.with_header_hide if headers and fmt.with_header_hide else []
    pad = fmt.padding
    headerrow = fmt.headerrow
    padded_widths = [w + 2 * pad for w in colwidths]
    if is_multiline:
        pad_row = lambda row, _: row
        append_row = partial(_append_multiline_row, pad=pad)
    else:
        pad_row = _pad_row
        append_row = _append_basic_row
    padded_headers = pad_row(headers, pad)
    padded_rows = [pad_row(row, pad) for row in rows]
    if fmt.lineabove and 'lineabove' not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.lineabove)
    if padded_headers:
        append_row(lines, padded_headers, padded_widths, colaligns, headerrow)
        if fmt.linebelowheader and 'linebelowheader' not in hidden:
            _append_line(lines, padded_widths, colaligns, fmt.linebelowheader)
    if padded_rows and fmt.linebetweenrows and ('linebetweenrows' not in hidden):
        for row, ralign in zip(padded_rows[:-1], rowaligns):
            append_row(lines, row, padded_widths, colaligns, fmt.datarow, rowalign=ralign)
            _append_line(lines, padded_widths, colaligns, fmt.linebetweenrows)
        append_row(lines, padded_rows[-1], padded_widths, colaligns, fmt.datarow, rowalign=rowaligns[-1])
    else:
        separating_line = fmt.linebetweenrows or fmt.linebelowheader or fmt.linebelow or fmt.lineabove or Line('', '', '', '')
        for row in padded_rows:
            if _is_separating_line(row):
                _append_line(lines, padded_widths, colaligns, separating_line)
            else:
                append_row(lines, row, padded_widths, colaligns, fmt.datarow)
    if fmt.linebelow and 'linebelow' not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.linebelow)
    if headers or rows:
        output = '\n'.join(lines)
        if fmt.lineabove == _html_begin_table_without_header:
            return JupyterHTMLStr(output)
        else:
            return output
    else:
        return ''