from __future__ import annotations
from ruamel.yaml.compat import nprintf  # NOQA
from .error import StreamMark  # NOQA
def add_comment_post(self, comment: Any) -> None:
    if not hasattr(self, '_comment'):
        self._comment = [None, None, []]
    else:
        assert len(self._comment) == 3
        if self._comment[2] is None:
            self._comment[2] = []
    self._comment[2].append(comment)