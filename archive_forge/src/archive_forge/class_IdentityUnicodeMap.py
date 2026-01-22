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
class IdentityUnicodeMap(UnicodeMap):

    def get_unichr(self, cid: int) -> str:
        """Interpret character id as unicode codepoint"""
        log.debug('get_unichr: %r, %r', self, cid)
        return chr(cid)