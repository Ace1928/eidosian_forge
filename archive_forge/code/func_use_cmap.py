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
def use_cmap(self, cmap: CMapBase) -> None:
    assert isinstance(cmap, CMap), str(type(cmap))

    def copy(dst: Dict[int, object], src: Dict[int, object]) -> None:
        for k, v in src.items():
            if isinstance(v, dict):
                d: Dict[int, object] = {}
                dst[k] = d
                copy(d, v)
            else:
                dst[k] = v
    copy(self.code2cid, cmap.code2cid)