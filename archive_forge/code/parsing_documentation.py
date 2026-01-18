import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
Appends a paragraph to the file

        >>> deb822_file = Deb822FileElement.new_empty_file()
        >>> para1 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para1["Source"] = "foo"
        >>> para1["Build-Depends"] = "debhelper-compat (= 13)"
        >>> para2 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para2["Package"] = "foo"
        >>> para2["Depends"] = "${shlib:Depends}, ${misc:Depends}"
        >>> deb822_file.append(para1)
        >>> deb822_file.append(para2)
        >>> expected = '''
        ... Source: foo
        ... Build-Depends: debhelper-compat (= 13)
        ...
        ... Package: foo
        ... Depends: ${shlib:Depends}, ${misc:Depends}
        ... '''.lstrip()
        >>> deb822_file.dump() == expected
        True
        