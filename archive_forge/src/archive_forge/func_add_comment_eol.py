from __future__ import annotations
from ruamel.yaml.compat import nprintf  # NOQA
from .error import StreamMark  # NOQA
def add_comment_eol(self, comment: Any, comment_type: Any) -> None:
    if not hasattr(self, '_comment'):
        self._comment = [None, None, None]
    else:
        assert len(self._comment) == 3
        assert self._comment[1] is None
    if self.comment[1] is None:
        self._comment[1] = []
    self._comment[1].extend([None] * (comment_type + 1 - len(self.comment[1])))
    self._comment[1][comment_type] = comment