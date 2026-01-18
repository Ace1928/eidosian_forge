from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
def tohtml(table, source=None, encoding=None, errors='strict', caption=None, vrepr=text_type, lineterminator='\n', index_header=False, tr_style=None, td_styles=None, truncate=None):
    """
    Write the table as HTML to a file. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> etl.tohtml(table1, 'example.html', caption='example table')
        >>> print(open('example.html').read())
        <table class='petl'>
        <caption>example table</caption>
        <thead>
        <tr>
        <th>foo</th>
        <th>bar</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>a</td>
        <td style='text-align: right'>1</td>
        </tr>
        <tr>
        <td>b</td>
        <td style='text-align: right'>2</td>
        </tr>
        <tr>
        <td>c</td>
        <td style='text-align: right'>2</td>
        </tr>
        </tbody>
        </table>

    The `caption` keyword argument is used to provide a table caption
    in the output HTML.

    """
    source = write_source_from_arg(source)
    with source.open('wb') as buf:
        if PY2:
            codec = getcodec(encoding)
            f = codec.streamwriter(buf, errors=errors)
        else:
            f = io.TextIOWrapper(buf, encoding=encoding, errors=errors, newline='')
        try:
            it = iter(table)
            try:
                hdr = next(it)
            except StopIteration:
                hdr = []
            _write_begin(f, hdr, lineterminator, caption, index_header, truncate)
            if tr_style and callable(tr_style):
                it = (Record(row, hdr) for row in it)
            for row in it:
                _write_row(f, hdr, row, lineterminator, vrepr, tr_style, td_styles, truncate)
            _write_end(f, lineterminator)
            f.flush()
        finally:
            if not PY2:
                f.detach()