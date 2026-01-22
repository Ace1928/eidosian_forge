from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def add_full_line_comment(self, comment: Any, column: Any, line: Any) -> Any:
    assert comment.count('\n') == 1 and comment[-1] == '\n'
    self.comments[line] = retval = FullLineComment(comment[:-1], line, column)
    self.unused.append(line)
    return retval