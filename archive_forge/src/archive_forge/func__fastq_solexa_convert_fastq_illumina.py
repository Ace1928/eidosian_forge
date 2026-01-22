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
def _fastq_solexa_convert_fastq_illumina(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    """Fast Solexa FASTQ to Illumina 1.3+ FASTQ conversion (PRIVATE).

    Avoids creating SeqRecord and Seq objects in order to speed up this
    conversion.
    """
    mapping = ''.join([chr(0) for ascii in range(59)] + [chr(64 + int(round(phred_quality_from_solexa(q)))) for q in range(-5, 62 + 1)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)