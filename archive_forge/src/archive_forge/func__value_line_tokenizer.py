import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
def _value_line_tokenizer(func):

    def impl(v):
        first_line = True
        for no, line in enumerate(v.splitlines(keepends=True)):
            assert not v.isspace() or no == 0
            if line.startswith('#'):
                yield Deb822CommentToken(line)
                continue
            has_newline = False
            continuation_line_marker = None
            if not first_line:
                continuation_line_marker = line[0]
                line = line[1:]
            first_line = False
            if line.endswith('\n'):
                has_newline = True
                line = line[:-1]
            if continuation_line_marker is not None:
                yield Deb822ValueContinuationToken(sys.intern(continuation_line_marker))
            yield from func(line)
            if has_newline:
                yield Deb822NewlineAfterValueToken()
    return impl