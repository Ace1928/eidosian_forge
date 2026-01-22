import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class FastaM10Indexer(SearchIndexer):
    """Indexer class for Bill Pearson's FASTA suite's -m 10 output."""
    _parser = FastaM10Parser

    def __init__(self, filename):
        """Initialize the class."""
        SearchIndexer.__init__(self, filename)

    def __iter__(self):
        """Iterate over FastaM10Indexer; yields query results' keys, start offsets, offset lengths."""
        handle = self._handle
        handle.seek(0)
        start_offset = handle.tell()
        qresult_key = None
        query_mark = b'>>>'
        line = handle.readline()
        while True:
            end_offset = handle.tell()
            if not line.startswith(query_mark) and query_mark in line:
                regx = re.search(_RE_ID_DESC_SEQLEN_IDX, line)
                qresult_key = regx.group(1).decode()
                start_offset = end_offset - len(line)
            if qresult_key is not None:
                if not line:
                    yield (qresult_key, start_offset, end_offset - start_offset)
                    break
                line = handle.readline()
                if not line.startswith(query_mark) and query_mark in line:
                    yield (qresult_key, start_offset, end_offset - start_offset)
                    start_offset = end_offset
            else:
                line = handle.readline()

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        qresult_raw = b''
        query_mark = b'>>>'
        handle.seek(0)
        line = handle.readline()
        while True:
            qresult_raw += line
            line = handle.readline()
            if not line.startswith(query_mark) and query_mark in line:
                break
        handle.seek(offset)
        line = handle.readline()
        while True:
            if not line:
                break
            qresult_raw += line
            line = handle.readline()
            if not line.startswith(query_mark) and query_mark in line:
                break
        return qresult_raw + b'>>><<<\n'