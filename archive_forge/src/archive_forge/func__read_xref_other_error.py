import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def _read_xref_other_error(self, stream: StreamType, startxref: int) -> Optional[int]:
    if startxref == 0:
        if self.strict:
            raise PdfReadError('/Prev=0 in the trailer (try opening with strict=False)')
        logger_warning('/Prev=0 in the trailer - assuming there is no previous xref table', __name__)
        return None
    stream.seek(-11, 1)
    tmp = stream.read(20)
    xref_loc = tmp.find(b'xref')
    if xref_loc != -1:
        startxref -= 10 - xref_loc
        return startxref
    stream.seek(startxref, 0)
    for look in range(25):
        if stream.read(1).isdigit():
            startxref += look
            return startxref
    if '/Root' in self.trailer and (not self.strict):
        logger_warning('Invalid parent xref., rebuild xref', __name__)
        try:
            self._rebuild_xref_table(stream)
            return None
        except Exception:
            raise PdfReadError('can not rebuild xref')
    raise PdfReadError('Could not find xref table at specified location')