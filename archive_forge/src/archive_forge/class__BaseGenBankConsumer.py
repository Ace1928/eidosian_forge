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
class _BaseGenBankConsumer:
    """Abstract GenBank consumer providing useful general functions (PRIVATE).

    This just helps to eliminate some duplication in things that most
    GenBank consumers want to do.
    """
    remove_space_keys = ['translation']

    def __init__(self):
        pass

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

    @staticmethod
    def _split_accessions(accession_string):
        """Split a string of accession numbers into a list (PRIVATE)."""
        accession = accession_string.replace('\n', ' ').replace(';', ' ')
        return [x.strip() for x in accession.split() if x.strip()]

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

    @staticmethod
    def _clean_location(location_string):
        """Clean whitespace out of a location string (PRIVATE).

        The location parser isn't a fan of whitespace, so we clean it out
        before feeding it into the parser.
        """
        return ''.join(location_string.split())

    @staticmethod
    def _remove_newlines(text):
        """Remove any newlines in the passed text, returning the new string (PRIVATE)."""
        newlines = ['\n', '\r']
        for ws in newlines:
            text = text.replace(ws, '')
        return text

    @staticmethod
    def _normalize_spaces(text):
        """Replace multiple spaces in the passed text with single spaces (PRIVATE)."""
        return ' '.join((x for x in text.split(' ') if x))

    @staticmethod
    def _remove_spaces(text):
        """Remove all spaces from the passed text (PRIVATE)."""
        return text.replace(' ', '')

    @staticmethod
    def _convert_to_python_numbers(start, end):
        """Convert a start and end range to python notation (PRIVATE).

        In GenBank, starts and ends are defined in "biological" coordinates,
        where 1 is the first base and [i, j] means to include both i and j.

        In python, 0 is the first base and [i, j] means to include i, but
        not j.

        So, to convert "biological" to python coordinates, we need to
        subtract 1 from the start, and leave the end and things should
        be converted happily.
        """
        new_start = start - 1
        new_end = end
        return (new_start, new_end)