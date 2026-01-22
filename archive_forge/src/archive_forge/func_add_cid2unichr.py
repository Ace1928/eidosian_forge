import gzip
import logging
import os
import os.path
import pickle as pickle
import struct
import sys
from typing import (
from .encodingdb import name2unicode
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import PSSyntaxError
from .psparser import literal_name
from .utils import choplist
from .utils import nunpack
def add_cid2unichr(self, cid: int, code: Union[PSLiteral, bytes, int]) -> None:
    assert isinstance(cid, int), str(type(cid))
    if isinstance(code, PSLiteral):
        assert isinstance(code.name, str)
        unichr = name2unicode(code.name)
    elif isinstance(code, bytes):
        unichr = code.decode('UTF-16BE', 'ignore')
    elif isinstance(code, int):
        unichr = chr(code)
    else:
        raise TypeError(code)
    if unichr == '\xa0' and self.cid2unichr.get(cid) == ' ':
        return
    self.cid2unichr[cid] = unichr