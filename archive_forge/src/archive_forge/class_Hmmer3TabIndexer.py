from itertools import chain
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class Hmmer3TabIndexer(SearchIndexer):
    """Indexer class for HMMER table output."""
    _parser = Hmmer3TabParser
    _query_id_idx = 2

    def __iter__(self):
        """Iterate over the file handle; yields key, start offset, and length."""
        handle = self._handle
        handle.seek(0)
        query_id_idx = self._query_id_idx
        qresult_key = None
        header_mark = b'#'
        split_mark = b' '
        line = header_mark
        while line.startswith(header_mark):
            start_offset = handle.tell()
            line = handle.readline()
        while True:
            end_offset = handle.tell()
            if not line:
                break
            cols = [x for x in line.strip().split(split_mark) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    adj_end = end_offset - len(line)
                    yield (qresult_key.decode(), start_offset, adj_end - start_offset)
                    qresult_key = curr_key
                    start_offset = adj_end
            line = handle.readline()
            if not line:
                yield (qresult_key.decode(), start_offset, end_offset - start_offset)
                break

    def get_raw(self, offset):
        """Return the raw bytes string of a QueryResult object from the given offset."""
        handle = self._handle
        handle.seek(offset)
        query_id_idx = self._query_id_idx
        qresult_key = None
        qresult_raw = b''
        split_mark = b' '
        while True:
            line = handle.readline()
            if not line:
                break
            cols = [x for x in line.strip().split(split_mark) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    break
            qresult_raw += line
        return qresult_raw