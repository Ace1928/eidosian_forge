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
@staticmethod
def _get_xref_issues(stream: StreamType, startxref: int) -> int:
    """
        Return an int which indicates an issue. 0 means there is no issue.

        Args:
            stream:
            startxref:

        Returns:
            0 means no issue, other values represent specific issues.
        """
    stream.seek(startxref - 1, 0)
    line = stream.read(1)
    if line == b'j':
        line = stream.read(1)
    if line not in b'\r\n \t':
        return 1
    line = stream.read(4)
    if line != b'xref':
        line = b''
        while line in b'0123456789 \t':
            line = stream.read(1)
            if line == b'':
                return 2
        line += stream.read(2)
        if line.lower() != b'obj':
            return 3
    return 0