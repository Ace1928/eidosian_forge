from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def add_blank_line(self, comment: Any, column: Any, line: Any) -> Any:
    assert comment.count('\n') == 1 and comment[-1] == '\n'
    assert line not in self.comments
    self.comments[line] = retval = BlankLineComment(comment[:-1], line, column)
    self.unused.append(line)
    return retval