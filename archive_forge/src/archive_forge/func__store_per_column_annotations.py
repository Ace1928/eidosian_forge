import textwrap
from collections import defaultdict
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _store_per_column_annotations(alignment, gc, columns, skipped_columns):
    if gc:
        alignment.column_annotations = {}
        for key, value in gc.items():
            if skipped_columns:
                value = ''.join((letter for index, letter in enumerate(value) if index not in skipped_columns))
            if len(value) != columns:
                raise ValueError(f'{key} length is {len(value)}, expected {columns}')
            alignment.column_annotations[AlignmentIterator.gc_mapping.get(key, key)] = value