import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
class Hmmer2TextIndexer(_BaseHmmerTextIndexer):
    """Indexer for hmmer2-text format."""
    _parser = Hmmer2TextParser
    qresult_start = b'Query'
    qresult_end = b'//'

    def __iter__(self):
        """Iterate over Hmmer2TextIndexer; yields query results' key, offsets, 0."""
        handle = self._handle
        handle.seek(0)
        start_offset = handle.tell()
        regex_id = re.compile(b'Query\\s*(?:sequence|HMM)?:\\s*(.*)')
        is_hmmsearch = False
        line = read_forward(handle)
        if line.startswith(b'hmmsearch'):
            is_hmmsearch = True
        while True:
            end_offset = handle.tell()
            if line.startswith(self.qresult_start):
                regx = re.search(regex_id, line)
                qresult_key = regx.group(1).strip()
                start_offset = end_offset - len(line)
            elif line.startswith(self.qresult_end):
                yield (qresult_key.decode(), start_offset, 0)
                start_offset = end_offset
            elif not line:
                if is_hmmsearch:
                    yield (qresult_key.decode(), start_offset, 0)
                break
            line = read_forward(handle)