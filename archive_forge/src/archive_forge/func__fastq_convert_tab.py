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
def _fastq_convert_tab(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    """Fast FASTQ to simple tabbed conversion (PRIVATE).

    Avoids dealing with the FASTQ quality encoding, and creating SeqRecord and
    Seq objects in order to speed up this conversion.

    NOTE - This does NOT check the characters used in the FASTQ quality string
    are valid!
    """
    count = 0
    with as_handle(out_file, 'w') as out_handle:
        for title, seq, qual in FastqGeneralIterator(in_file):
            count += 1
            out_handle.write(f'{title.split(None, 1)[0]}\t{seq}\n')
    return count