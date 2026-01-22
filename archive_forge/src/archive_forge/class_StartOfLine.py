import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class StartOfLine(ZeroWidthBase):
    _opcode = OP.START_OF_LINE
    _op_name = 'START_OF_LINE'