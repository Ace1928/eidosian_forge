import logging
from io import BytesIO
from typing import BinaryIO, TYPE_CHECKING, Optional, Union
from . import settings
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSStackParser
from .psparser import PSSyntaxError
class PDFStreamParser(PDFParser):
    """
    PDFStreamParser is used to parse PDF content streams
    that is contained in each page and has instructions
    for rendering the page. A reference to a PDF document is
    needed because a PDF content stream can also have
    indirect references to other objects in the same document.
    """

    def __init__(self, data: bytes) -> None:
        PDFParser.__init__(self, BytesIO(data))

    def flush(self) -> None:
        self.add_results(*self.popall())
    KEYWORD_OBJ = KWD(b'obj')

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_R:
            try:
                (_, objid), (_, genno) = self.pop(2)
                objid, genno = (int(objid), int(genno))
                obj = PDFObjRef(self.doc, objid, genno)
                self.push((pos, obj))
            except PSSyntaxError:
                pass
            return
        elif token in (self.KEYWORD_OBJ, self.KEYWORD_ENDOBJ):
            if settings.STRICT:
                raise PDFSyntaxError('Keyword endobj found in stream')
            return
        self.push((pos, token))