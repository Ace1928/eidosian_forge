import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
class ExonerateTextIndexer(_BaseExonerateIndexer):
    """Indexer class for Exonerate plain text."""
    _parser = ExonerateTextParser
    _query_mark = b'C4 Alignment'

    def get_qresult_id(self, pos):
        """Return the query ID from the nearest "Query:" line."""
        handle = self._handle
        handle.seek(pos)
        sentinel = b'Query:'
        while True:
            line = handle.readline().strip()
            if line.startswith(sentinel):
                break
            if not line:
                raise StopIteration
        qid, desc = _parse_hit_or_query_line(line.decode())
        return qid

    def get_raw(self, offset):
        """Return the raw string of a QueryResult object from the given offset."""
        handle = self._handle
        handle.seek(offset)
        qresult_key = None
        qresult_raw = b''
        while True:
            line = handle.readline()
            if not line:
                break
            elif line.startswith(self._query_mark):
                cur_pos = handle.tell()
                if qresult_key is None:
                    qresult_key = self.get_qresult_id(cur_pos)
                else:
                    curr_key = self.get_qresult_id(cur_pos)
                    if curr_key != qresult_key:
                        break
                handle.seek(cur_pos)
            qresult_raw += line
        return qresult_raw