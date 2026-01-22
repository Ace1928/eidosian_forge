import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def MafIterator(handle, seq_count=None):
    """Iterate over a MAF file handle as MultipleSeqAlignment objects.

    Iterates over lines in a MAF file-like object (handle), yielding
    MultipleSeqAlignment objects. SeqRecord IDs generally correspond to
    species names.
    """
    in_a_bundle = False
    annotations = []
    records = []
    while True:
        try:
            line = next(handle)
        except StopIteration:
            line = ''
        if in_a_bundle:
            if line.startswith('s'):
                line_split = line.strip().split()
                if len(line_split) != 7:
                    raise ValueError("Error parsing alignment - 's' line must have 7 fields")
                if line_split[4] == '+':
                    strand = 1
                elif line_split[4] == '-':
                    strand = -1
                else:
                    strand = 1
                anno = {'start': int(line_split[2]), 'size': int(line_split[3]), 'strand': strand, 'srcSize': int(line_split[5])}
                sequence = line_split[6]
                if '.' in sequence:
                    if not records:
                        raise ValueError('Found dot/period in first sequence of alignment')
                    ref = records[0].seq
                    new = []
                    for letter, ref_letter in zip(sequence, ref):
                        new.append(ref_letter if letter == '.' else letter)
                    sequence = ''.join(new)
                records.append(SeqRecord(Seq(sequence), id=line_split[1], name=line_split[1], description='', annotations=anno))
            elif line.startswith('i'):
                pass
            elif line.startswith('e'):
                pass
            elif line.startswith('q'):
                pass
            elif line.startswith('#'):
                pass
            elif not line.strip():
                if seq_count is not None:
                    assert len(records) == seq_count
                alignment = MultipleSeqAlignment(records)
                alignment._annotations = annotations
                yield alignment
                in_a_bundle = False
                annotations = []
                records = []
            else:
                raise ValueError(f'Error parsing alignment - unexpected line:\n{line}')
        elif line.startswith('a'):
            in_a_bundle = True
            annot_strings = line.strip().split()[1:]
            if len(annot_strings) != line.count('='):
                raise ValueError("Error parsing alignment - invalid key in 'a' line")
            annotations = dict((a_string.split('=') for a_string in annot_strings))
        elif line.startswith('#'):
            pass
        elif not line:
            break