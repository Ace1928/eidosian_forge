import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _insdc_feature_position_string(pos, offset=0):
    """Build a GenBank/EMBL position string (PRIVATE).

    Use offset=1 to add one to convert a start position from python counting.
    """
    if isinstance(pos, SeqFeature.ExactPosition):
        return '%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.WithinPosition):
        return '(%i.%i)' % (pos._left + offset, pos._right + offset)
    elif isinstance(pos, SeqFeature.BetweenPosition):
        return '(%i^%i)' % (pos._left + offset, pos._right + offset)
    elif isinstance(pos, SeqFeature.BeforePosition):
        return '<%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.AfterPosition):
        return '>%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.OneOfPosition):
        return 'one-of(%s)' % ','.join((_insdc_feature_position_string(p, offset) for p in pos.position_choices))
    elif isinstance(pos, SeqFeature.Position):
        raise NotImplementedError('Please report this as a bug in Biopython.')
    else:
        raise ValueError('Expected a SeqFeature position object.')