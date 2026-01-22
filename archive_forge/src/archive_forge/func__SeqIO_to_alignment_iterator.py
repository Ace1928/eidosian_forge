from Bio.Align import MultipleSeqAlignment
from Bio.File import as_handle
from . import ClustalIO
from . import EmbossIO
from . import FastaIO
from . import MafIO
from . import MauveIO
from . import MsfIO
from . import NexusIO
from . import PhylipIO
from . import StockholmIO
def _SeqIO_to_alignment_iterator(handle, format, seq_count=None):
    """Use Bio.SeqIO to create an MultipleSeqAlignment iterator (PRIVATE).

    Arguments:
     - handle    - handle to the file.
     - format    - string describing the file format.
     - seq_count - Optional integer, number of sequences expected in each
       alignment.  Recommended for fasta format files.

    If count is omitted (default) then all the sequences in the file are
    combined into a single MultipleSeqAlignment.
    """
    from Bio import SeqIO
    if format not in SeqIO._FormatToIterator:
        raise ValueError(f"Unknown format '{format}'")
    if seq_count:
        seq_record_iterator = SeqIO.parse(handle, format)
        records = []
        for record in seq_record_iterator:
            records.append(record)
            if len(records) == seq_count:
                yield MultipleSeqAlignment(records)
                records = []
        if records:
            raise ValueError('Check seq_count argument, not enough sequences?')
    else:
        records = list(SeqIO.parse(handle, format))
        if records:
            yield MultipleSeqAlignment(records)