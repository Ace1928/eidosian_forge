import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class EndOfLineU(EndOfLine):
    _opcode = OP.END_OF_LINE_U
    _op_name = 'END_OF_LINE_U'