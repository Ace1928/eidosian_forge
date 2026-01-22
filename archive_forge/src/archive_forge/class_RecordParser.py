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
class RecordParser:
    """Parse GenBank files into Record objects (OBSOLETE).

    Direct use of this class is discouraged, and may be deprecated in
    a future release of Biopython.

    Please use the Bio.GenBank.parse(...) or Bio.GenBank.read(...) functions
    instead.
    """

    def __init__(self, debug_level=0):
        """Initialize the parser.

        Arguments:
         - debug_level - An optional argument that species the amount of
           debugging information the parser should spit out. By default we have
           no debugging info (the fastest way to do things), but if you want
           you can set this as high as two and see exactly where a parse fails.

        """
        self._scanner = GenBankScanner(debug_level)

    def parse(self, handle):
        """Parse the specified handle into a GenBank record."""
        _consumer = _RecordConsumer()
        self._scanner.feed(handle, _consumer)
        return _consumer.data