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
def _split_reference_locations(self, location_string):
    """Get reference locations out of a string of reference information (PRIVATE).

        The passed string should be of the form::

            1 to 20; 20 to 100

        This splits the information out and returns a list of location objects
        based on the reference locations.
        """
    all_base_info = location_string.split(';')
    new_locations = []
    for base_info in all_base_info:
        start, end = base_info.split('to')
        new_start, new_end = self._convert_to_python_numbers(int(start.strip()), int(end.strip()))
        this_location = SimpleLocation(new_start, new_end)
        new_locations.append(this_location)
    return new_locations