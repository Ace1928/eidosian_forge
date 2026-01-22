import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Keep(ZeroWidthBase):
    _opcode = OP.KEEP
    _op_name = 'KEEP'