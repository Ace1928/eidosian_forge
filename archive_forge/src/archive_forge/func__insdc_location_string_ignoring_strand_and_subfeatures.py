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
def _insdc_location_string_ignoring_strand_and_subfeatures(location, rec_length):
    if location.ref:
        ref = f'{location.ref}:'
    else:
        ref = ''
    assert not location.ref_db
    if isinstance(location.start, SeqFeature.ExactPosition) and isinstance(location.end, SeqFeature.ExactPosition) and (location.start == location.end):
        if location.end == rec_length:
            return '%s%i^1' % (ref, rec_length)
        else:
            return '%s%i^%i' % (ref, location.end, location.end + 1)
    if isinstance(location.start, SeqFeature.ExactPosition) and isinstance(location.end, SeqFeature.ExactPosition) and (location.start + 1 == location.end):
        return '%s%i' % (ref, location.end)
    elif isinstance(location.start, SeqFeature.UnknownPosition) or isinstance(location.end, SeqFeature.UnknownPosition):
        if isinstance(location.start, SeqFeature.UnknownPosition) and isinstance(location.end, SeqFeature.UnknownPosition):
            raise ValueError('Feature with unknown location')
        elif isinstance(location.start, SeqFeature.UnknownPosition):
            return '%s<%i..%s' % (ref, location.end, _insdc_feature_position_string(location.end))
        else:
            return '%s%s..>%i' % (ref, _insdc_feature_position_string(location.start, +1), location.start + 1)
    else:
        return ref + _insdc_feature_position_string(location.start, +1) + '..' + _insdc_feature_position_string(location.end)