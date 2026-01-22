import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
class PatienceSequenceMatcher_py(difflib.SequenceMatcher):
    """Compare a pair of sequences using longest common subset."""
    _do_check_consistency = True

    def __init__(self, isjunk=None, a='', b='') -> None:
        if isjunk is not None:
            raise NotImplementedError('Currently we do not support isjunk for sequence matching')
        difflib.SequenceMatcher.__init__(self, isjunk, a, b)

    def get_matching_blocks(self):
        """Return list of triples describing matching subsequences.

        Each triple is of the form (i, j, n), and means that
        a[i:i+n] == b[j:j+n].  The triples are monotonically increasing in
        i and in j.

        The last triple is a dummy, (len(a), len(b), 0), and is the only
        triple with n==0.

        >>> s = PatienceSequenceMatcher(None, "abxcd", "abcd")
        >>> s.get_matching_blocks()
        [(0, 0, 2), (3, 2, 2), (5, 4, 0)]
        """
        if self.matching_blocks is not None:
            return self.matching_blocks
        matches = []
        recurse_matches_py(self.a, self.b, 0, 0, len(self.a), len(self.b), matches, 10)
        self.matching_blocks = _collapse_sequences(matches)
        self.matching_blocks.append((len(self.a), len(self.b), 0))
        if PatienceSequenceMatcher_py._do_check_consistency:
            if __debug__:
                _check_consistency(self.matching_blocks)
        return self.matching_blocks