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
def _read_pdf15_xref_stream(self, stream: StreamType) -> Union[ContentStream, EncodedStreamObject, DecodedStreamObject]:
    stream.seek(-1, 1)
    idnum, generation = self.read_object_header(stream)
    xrefstream = cast(ContentStream, read_object(stream, self))
    assert cast(str, xrefstream['/Type']) == '/XRef'
    self.cache_indirect_object(generation, idnum, xrefstream)
    stream_data = BytesIO(b_(xrefstream.get_data()))
    idx_pairs = xrefstream.get('/Index', [0, xrefstream.get('/Size')])
    entry_sizes = cast(Dict[Any, Any], xrefstream.get('/W'))
    assert len(entry_sizes) >= 3
    if self.strict and len(entry_sizes) > 3:
        raise PdfReadError(f'Too many entry sizes: {entry_sizes}')

    def get_entry(i: int) -> Union[int, Tuple[int, ...]]:
        if entry_sizes[i] > 0:
            d = stream_data.read(entry_sizes[i])
            return convert_to_int(d, entry_sizes[i])
        if i == 0:
            return 1
        else:
            return 0

    def used_before(num: int, generation: Union[int, Tuple[int, ...]]) -> bool:
        return num in self.xref.get(generation, []) or num in self.xref_objStm
    self._read_xref_subsections(idx_pairs, get_entry, used_before)
    return xrefstream