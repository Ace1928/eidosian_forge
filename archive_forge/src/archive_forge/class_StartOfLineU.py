import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class StartOfLineU(StartOfLine):
    _opcode = OP.START_OF_LINE_U
    _op_name = 'START_OF_LINE_U'