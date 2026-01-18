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
def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
    if encryption_key is not None:
        deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
    stream.write(self.renumber())