from __future__ import annotations
import re
from . import Image, _imagingmorph
def _string_permute(self, pattern, permutation):
    """string_permute takes a pattern and a permutation and returns the
        string permuted according to the permutation list.
        """
    assert len(permutation) == 9
    return ''.join((pattern[p] for p in permutation))