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
def version_suffix(self, version):
    """Set the version to overwrite the id.

        Since the version provides the same information as the accession
        number, plus some extra info, we set this as the id if we have
        a version.
        """
    assert version.isdigit()
    self.data.annotations['sequence_version'] = int(version)