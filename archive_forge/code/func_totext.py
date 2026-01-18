from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
def totext(table, source=None, encoding=None, errors='strict', template=None, prologue=None, epilogue=None):
    """
    Write the table to a text file. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> prologue = '''{| class="wikitable"
        ... |-
        ... ! foo
        ... ! bar
        ... '''
        >>> template = '''|-
        ... | {foo}
        ... | {bar}
        ... '''
        >>> epilogue = '|}'
        >>> etl.totext(table1, 'example.txt', template=template,
        ... prologue=prologue, epilogue=epilogue)
        >>> # see what we did
        ... print(open('example.txt').read())
        {| class="wikitable"
        |-
        ! foo
        ! bar
        |-
        | a
        | 1
        |-
        | b
        | 2
        |-
        | c
        | 2
        |}

    The `template` will be used to format each row via
    `str.format <http://docs.python.org/library/stdtypes.html#str.format>`_.

    """
    _writetext(table, source=source, mode='wb', encoding=encoding, errors=errors, template=template, prologue=prologue, epilogue=epilogue)