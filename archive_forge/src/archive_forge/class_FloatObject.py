import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
class FloatObject(float, PdfObject):

    def __new__(cls, value: Union[str, Any]='0.0', context: Optional[Any]=None) -> 'FloatObject':
        try:
            value = float(str_(value))
            return float.__new__(cls, value)
        except Exception as e:
            logger_warning(f'{e} : FloatObject ({value}) invalid; use 0.0 instead', __name__)
            return float.__new__(cls, 0.0)

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'FloatObject':
        """Clone object into pdf_dest."""
        return cast('FloatObject', self._reference_clone(FloatObject(self), pdf_dest, force_duplicate))

    def myrepr(self) -> str:
        if self == 0:
            return '0.0'
        nb = FLOAT_WRITE_PRECISION - int(log10(abs(self)))
        s = f'{self:.{max(1, nb)}f}'.rstrip('0').rstrip('.')
        return s

    def __repr__(self) -> str:
        return self.myrepr()

    def as_numeric(self) -> float:
        return float(self)

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(self.myrepr().encode('utf8'))