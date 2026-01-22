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
def _split_taxonomy(taxonomy_string):
    """Split a string with taxonomy info into a list (PRIVATE)."""
    if not taxonomy_string or taxonomy_string == '.':
        return []
    if taxonomy_string[-1] == '.':
        tax_info = taxonomy_string[:-1]
    else:
        tax_info = taxonomy_string
    tax_list = tax_info.split(';')
    new_tax_list = []
    for tax_item in tax_list:
        new_items = tax_item.split('\n')
        new_tax_list.extend(new_items)
    while '' in new_tax_list:
        new_tax_list.remove('')
    return [x.strip() for x in new_tax_list]