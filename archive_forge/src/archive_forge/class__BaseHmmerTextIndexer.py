from Bio.SearchIO._index import SearchIndexer
class _BaseHmmerTextIndexer(SearchIndexer):
    """Base indexer class for HMMER plain text output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preamble = b''

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string."""
        handle = self._handle
        qresult_raw = b''
        if not self._preamble:
            handle.seek(0)
            while True:
                line = handle.readline()
                if line.startswith(self.qresult_start):
                    break
                qresult_raw += line
        else:
            qresult_raw += self._preamble
        handle.seek(offset)
        while True:
            line = handle.readline()
            qresult_raw += line
            if line.startswith(self.qresult_end) or not line:
                break
        return qresult_raw