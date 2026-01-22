import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _next_state(self, leaf: Leaf) -> bool:
    """
        Pre-conditions:
            * On the first call to this function, @leaf MUST be the leaf that
              was directly after the string leaf in question (e.g. if our target
              string is `line.leaves[i]` then the first call to this method must
              be `line.leaves[i + 1]`).
            * On the next call to this function, the leaf parameter passed in
              MUST be the leaf directly following @leaf.

        Returns:
            True iff @leaf is a part of the string's trailer.
        """
    if is_empty_par(leaf):
        return True
    next_token = leaf.type
    if next_token == token.LPAR:
        self._unmatched_lpars += 1
    current_state = self._state
    if current_state == self.LPAR:
        if next_token == token.RPAR:
            self._unmatched_lpars -= 1
            if self._unmatched_lpars == 0:
                self._state = self.RPAR
    else:
        if (current_state, next_token) in self._goto:
            self._state = self._goto[current_state, next_token]
        elif (current_state, self.DEFAULT_TOKEN) in self._goto:
            self._state = self._goto[current_state, self.DEFAULT_TOKEN]
        else:
            raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
        if self._state == self.DONE:
            return False
    return True