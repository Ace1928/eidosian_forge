import warnings
from re import match
from struct import pack
from struct import unpack
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _parse_feature_description(desc, qualifiers):
    """Parse the description field of a Xdna feature.

    The 'description' field of a feature sometimes contains several
    GenBank-like qualifiers, separated by carriage returns (CR, 0x0D).
    """
    for line in [x for x in desc.split('\r') if len(x) > 0]:
        m = match('^([^=]+)="([^"]+)"?$', line)
        if m:
            qual, value = m.groups()
            qualifiers[qual] = [value]
        elif '"' not in line:
            qualifiers['note'] = [line]