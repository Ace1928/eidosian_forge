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
def _fastq_sanger_convert_fastq_sanger(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    """Fast Sanger FASTQ to Sanger FASTQ conversion (PRIVATE).

    Useful for removing line wrapping and the redundant second identifier
    on the plus lines. Will check also check the quality string is valid.

    Avoids creating SeqRecord and Seq objects in order to speed up this
    conversion.
    """
    mapping = ''.join([chr(0) for ascii in range(33)] + [chr(ascii) for ascii in range(33, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)