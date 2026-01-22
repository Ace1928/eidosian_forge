import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _get_raw_qresult(self, offset):
    """Return the raw bytes string of a single QueryResult from a noncommented file (PRIVATE)."""
    handle = self._handle
    handle.seek(offset)
    qresult_raw = b''
    key_idx = self._key_idx
    qresult_key = None
    while True:
        line = handle.readline()
        if qresult_key is None:
            qresult_key = line.split(b'\t')[key_idx]
        else:
            try:
                curr_key = line.split(b'\t')[key_idx]
            except IndexError:
                curr_key = b''
            if curr_key != qresult_key:
                break
        qresult_raw += line
    return qresult_raw