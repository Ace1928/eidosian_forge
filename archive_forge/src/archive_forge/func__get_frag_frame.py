import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _get_frag_frame(self, frag, seq_type, parsedict):
    """Return fragment frame for given object (PRIVATE).

        Returns ``HSPFragment`` frame given the object, its sequence type,
        and its parsed dictionary values.
        """
    assert seq_type in ('query', 'hit')
    frame = getattr(frag, '%s_frame' % seq_type, None)
    if frame is not None:
        return frame
    elif 'frames' in parsedict:
        idx = 0 if seq_type == 'query' else 1
        return int(parsedict['frames'].split('/')[idx])