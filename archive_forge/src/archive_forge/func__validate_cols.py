import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _validate_cols(self, cols):
    """Validate column's length of PSL or PSLX (PRIVATE)."""
    if not self.pslx:
        if len(cols) != 21:
            raise ValueError('Invalid PSL line: %r. Expected 21 tab-separated columns, found %i' % (self.line, len(cols)))
    elif len(cols) != 23:
        raise ValueError('Invalid PSLX line: %r. Expected 23 tab-separated columns, found %i' % (self.line, len(cols)))