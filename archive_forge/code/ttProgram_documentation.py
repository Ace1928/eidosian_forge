from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging

        >>> p = Program()
        >>> bool(p)
        False
        >>> bc = array.array("B", [0])
        >>> p.fromBytecode(bc)
        >>> bool(p)
        True
        >>> p.bytecode.pop()
        0
        >>> bool(p)
        False

        >>> p = Program()
        >>> asm = ['SVTCA[0]']
        >>> p.fromAssembly(asm)
        >>> bool(p)
        True
        >>> p.assembly.pop()
        'SVTCA[0]'
        >>> bool(p)
        False
        