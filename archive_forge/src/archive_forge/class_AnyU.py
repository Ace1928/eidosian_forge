import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class AnyU(Any):
    _opcode = {False: OP.ANY_U, True: OP.ANY_U_REV}
    _op_name = 'ANY_U'