from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def unprocessed(self, use: Any=False) -> Any:
    while len(self.unused) > 0:
        if _debug != 0:
            import inspect
            first = self.unused.pop(0) if use else self.unused[0]
            info = inspect.getframeinfo(inspect.stack()[1][0])
            xprintf('using', first, self.comments[first].value, info.function, info.lineno)
        yield (first, self.comments[first])
        if use:
            self.comments[first].set_used()