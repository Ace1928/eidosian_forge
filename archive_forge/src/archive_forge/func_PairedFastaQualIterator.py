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
def PairedFastaQualIterator(fasta_source: _TextIOSource, qual_source: _TextIOSource, alphabet: None=None) -> Iterator[SeqRecord]:
    """Iterate over matched FASTA and QUAL files as SeqRecord objects.

    For example, consider this short QUAL file with PHRED quality scores::

        >EAS54_6_R1_2_1_413_324
        26 26 18 26 26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26
        26 26 26 23 23
        >EAS54_6_R1_2_1_540_792
        26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26 26 12 26 26
        26 18 26 23 18
        >EAS54_6_R1_2_1_443_348
        26 26 26 26 26 26 26 26 26 26 26 24 26 22 26 26 13 22 26 18
        24 18 18 18 18

    And a matching FASTA file::

        >EAS54_6_R1_2_1_413_324
        CCCTTCTTGTCTTCAGCGTTTCTCC
        >EAS54_6_R1_2_1_540_792
        TTGGCAGGCCAAGGCCGATGGATCA
        >EAS54_6_R1_2_1_443_348
        GTTGCTTCTGGCGTGGGTGGGGGGG

    You can parse these separately using Bio.SeqIO with the "qual" and
    "fasta" formats, but then you'll get a group of SeqRecord objects with
    no sequence, and a matching group with the sequence but not the
    qualities.  Because it only deals with one input file handle, Bio.SeqIO
    can't be used to read the two files together - but this function can!
    For example,

    >>> with open("Quality/example.fasta") as f:
    ...     with open("Quality/example.qual") as q:
    ...         for record in PairedFastaQualIterator(f, q):
    ...             print("%s %s" % (record.id, record.seq))
    ...
    EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC
    EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA
    EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG

    As with the FASTQ or QUAL parsers, if you want to look at the qualities,
    they are in each record's per-letter-annotation dictionary as a simple
    list of integers:

    >>> print(record.letter_annotations["phred_quality"])
    [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]

    If you have access to data as a FASTQ format file, using that directly
    would be simpler and more straight forward.  Note that you can easily use
    this function to convert paired FASTA and QUAL files into FASTQ files:

    >>> from Bio import SeqIO
    >>> with open("Quality/example.fasta") as f:
    ...     with open("Quality/example.qual") as q:
    ...         SeqIO.write(PairedFastaQualIterator(f, q), "Quality/temp.fastq", "fastq")
    ...
    3

    And don't forget to clean up the temp file if you don't need it anymore:

    >>> import os
    >>> os.remove("Quality/temp.fastq")
    """
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    from Bio.SeqIO.FastaIO import FastaIterator
    fasta_iter = FastaIterator(fasta_source)
    qual_iter = QualPhredIterator(qual_source)
    while True:
        try:
            f_rec = next(fasta_iter)
        except StopIteration:
            f_rec = None
        try:
            q_rec = next(qual_iter)
        except StopIteration:
            q_rec = None
        if f_rec is None and q_rec is None:
            break
        if f_rec is None:
            raise ValueError('FASTA file has more entries than the QUAL file.')
        if q_rec is None:
            raise ValueError('QUAL file has more entries than the FASTA file.')
        if f_rec.id != q_rec.id:
            raise ValueError(f'FASTA and QUAL entries do not match ({f_rec.id} vs {q_rec.id}).')
        if len(f_rec) != len(q_rec.letter_annotations['phred_quality']):
            raise ValueError(f'Sequence length and number of quality scores disagree for {f_rec.id}')
        f_rec.letter_annotations['phred_quality'] = q_rec.letter_annotations['phred_quality']
        yield f_rec