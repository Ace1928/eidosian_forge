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
def _get_phred_quality(record: SeqRecord) -> Union[List[float], List[int]]:
    """Extract PHRED qualities from a SeqRecord's letter_annotations (PRIVATE).

    If there are no PHRED qualities, but there are Solexa qualities, those are
    used instead after conversion.
    """
    try:
        return record.letter_annotations['phred_quality']
    except KeyError:
        pass
    try:
        return [phred_quality_from_solexa(q) for q in record.letter_annotations['solexa_quality']]
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None