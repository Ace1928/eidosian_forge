import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _qresult_index_commented(self):
    """Indexer for commented BLAST tabular files (PRIVATE)."""
    handle = self._handle
    handle.seek(0)
    start_offset = 0
    query_mark = None
    qid_mark = b'# Query: '
    end_mark = b'# BLAST processed'
    while True:
        end_offset = handle.tell()
        line = handle.readline()
        if query_mark is None:
            query_mark = line
            start_offset = end_offset
        elif line.startswith(qid_mark):
            qresult_key = line[len(qid_mark):].split()[0]
        elif line == query_mark or line.startswith(end_mark):
            yield (qresult_key, start_offset, end_offset - start_offset)
            start_offset = end_offset
        elif not line:
            break