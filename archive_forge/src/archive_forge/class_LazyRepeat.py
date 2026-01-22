import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class LazyRepeat(GreedyRepeat):
    _opcode = OP.LAZY_REPEAT
    _op_name = 'LAZY_REPEAT'