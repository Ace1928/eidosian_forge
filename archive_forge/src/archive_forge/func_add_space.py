from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def add_space(letter):
    """Add a space before a capital letter."""
    if letter.isupper():
        return ' ' + letter.lower()
    else:
        return letter