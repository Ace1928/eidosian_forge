from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import (
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
def writeWithAttributeEscaping(write: Callable[[bytes], object]) -> Callable[[bytes], None]:
    """
    Decorate a C{write} callable so that all output written is properly quoted
    for inclusion within an XML attribute value.

    If a L{Tag <twisted.web.template.Tag>} C{x} is flattened within the context
    of the contents of another L{Tag <twisted.web.template.Tag>} C{y}, the
    metacharacters (C{<>&"}) delimiting C{x} should be passed through
    unchanged, but the textual content of C{x} should still be quoted, as
    usual.  For example: C{<y><x>&amp;</x></y>}.  That is the default behavior
    of L{_flattenElement} when L{escapeForContent} is passed as the
    C{dataEscaper}.

    However, when a L{Tag <twisted.web.template.Tag>} C{x} is flattened within
    the context of an I{attribute} of another L{Tag <twisted.web.template.Tag>}
    C{y}, then the metacharacters delimiting C{x} should be quoted so that it
    can be parsed from the attribute's value.  In the DOM itself, this is not a
    valid thing to do, but given that renderers and slots may be freely moved
    around in a L{twisted.web.template} template, it is a condition which may
    arise in a document and must be handled in a way which produces valid
    output.  So, for example, you should be able to get C{<y attr="&lt;x /&gt;"
    />}.  This should also be true for other XML/HTML meta-constructs such as
    comments and CDATA, so if you were to serialize a L{comment
    <twisted.web.template.Comment>} in an attribute you should get C{<y
    attr="&lt;-- comment --&gt;" />}.  Therefore in order to capture these
    meta-characters, flattening is done with C{write} callable that is wrapped
    with L{writeWithAttributeEscaping}.

    The final case, and hopefully the much more common one as compared to
    serializing L{Tag <twisted.web.template.Tag>} and arbitrary L{IRenderable}
    objects within an attribute, is to serialize a simple string, and those
    should be passed through for L{writeWithAttributeEscaping} to quote
    without applying a second, redundant level of quoting.

    @param write: A callable which will be invoked with the escaped L{bytes}.

    @return: A callable that writes data with escaping.
    """

    def _write(data: bytes) -> None:
        write(escapeForContent(data).replace(b'"', b'&quot;'))
    return _write