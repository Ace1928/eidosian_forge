import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
@staticmethod
def _split_keywords(keyword_string):
    """Split a string of keywords into a nice clean list (PRIVATE)."""
    if keyword_string == '' or keyword_string == '.':
        keywords = ''
    elif keyword_string[-1] == '.':
        keywords = keyword_string[:-1]
    else:
        keywords = keyword_string
    keyword_list = keywords.split(';')
    return [x.strip() for x in keyword_list]