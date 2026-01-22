from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_seqfeature_locations(self, feature, seqfeature_id):
    """Load all of the locations for a SeqFeature into tables (PRIVATE).

        This adds the locations related to the SeqFeature into the
        seqfeature_location table. Fuzzies are not handled right now.
        For a simple location, ie (1..2), we have a single table row
        with seq_start = 1, seq_end = 2, location_rank = 1.

        For split locations, ie (1..2, 3..4, 5..6) we would have three
        row tables with::

            start = 1, end = 2, rank = 1
            start = 3, end = 4, rank = 2
            start = 5, end = 6, rank = 3

        """
    try:
        if feature.location.operator != 'join':
            import warnings
            from Bio import BiopythonWarning
            warnings.warn('%s location operators are not fully supported' % feature.location_operator, BiopythonWarning)
    except AttributeError:
        pass
    parts = feature.location.parts
    if parts and {loc.strand for loc in parts} == {-1}:
        parts = parts[::-1]
    for rank, loc in enumerate(parts):
        self._insert_location(loc, rank + 1, seqfeature_id)