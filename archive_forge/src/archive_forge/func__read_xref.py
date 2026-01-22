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
def _read_xref(self, stream: StreamType) -> Optional[int]:
    self._read_standard_xref_table(stream)
    if stream.read(1) == b'':
        return None
    stream.seek(-1, 1)
    read_non_whitespace(stream)
    stream.seek(-1, 1)
    new_trailer = cast(Dict[str, Any], read_object(stream, self))
    for key, value in new_trailer.items():
        if key not in self.trailer:
            self.trailer[key] = value
    if '/XRefStm' in new_trailer:
        p = stream.tell()
        stream.seek(cast(int, new_trailer['/XRefStm']) + 1, 0)
        try:
            self._read_pdf15_xref_stream(stream)
        except Exception:
            logger_warning(f'XRef object at {new_trailer['/XRefStm']} can not be read, some object may be missing', __name__)
        stream.seek(p, 0)
    if '/Prev' in new_trailer:
        startxref = new_trailer['/Prev']
        return startxref
    else:
        return None