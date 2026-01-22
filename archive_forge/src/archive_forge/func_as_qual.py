import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
def as_qual(record: SeqRecord) -> str:
    """Turn a SeqRecord into a QUAL formatted string.

    This is used internally by the SeqRecord's .format("qual")
    method and by the SeqIO.write(..., ..., "qual") function.
    """
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    lines = [f'>{title}\n']
    qualities = _get_phred_quality(record)
    try:
        qualities_strs = ['%i' % round(q, 0) for q in qualities]
    except TypeError:
        if None in qualities:
            raise TypeError('A quality value of None was found') from None
        else:
            raise
    while qualities_strs:
        line = qualities_strs.pop(0)
        while qualities_strs and len(line) + 1 + len(qualities_strs[0]) < 60:
            line += ' ' + qualities_strs.pop(0)
        lines.append(line + '\n')
    return ''.join(lines)