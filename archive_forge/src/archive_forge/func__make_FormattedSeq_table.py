import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
def _make_FormattedSeq_table() -> bytes:
    table = bytearray(256)
    upper_to_lower = ord('A') - ord('a')
    for c in b'ABCDGHKMNRSTVWY':
        table[c] = c
        table[c - upper_to_lower] = c
    return bytes(table)