import collections
import re
import uuid
from ply import lex
from ply import yacc
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import lexer
from yaql.language import parser
from yaql.language import utils
@staticmethod
def _name_generator():
    value = 1
    while True:
        t = value
        chars = []
        while t:
            chars.append(chr(ord('A') + t % 26))
            t //= 26
        yield ''.join(chars)
        value += 1