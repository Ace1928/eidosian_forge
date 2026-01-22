import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _qresult_index(self):
    """Indexer for noncommented BLAST tabular files (PRIVATE)."""
    handle = self._handle
    handle.seek(0)
    start_offset = 0
    qresult_key = None
    key_idx = self._key_idx
    while True:
        end_offset = handle.tell()
        line = handle.readline()
        if qresult_key is None:
            qresult_key = line.split(b'\t')[key_idx]
        else:
            try:
                curr_key = line.split(b'\t')[key_idx]
            except IndexError:
                curr_key = b''
            if curr_key != qresult_key:
                yield (qresult_key, start_offset, end_offset - start_offset)
                qresult_key = curr_key
                start_offset = end_offset
        if not line:
            break