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
class YaqlOperators(object):

    def __init__(self, operators, name_value_op=None):
        self.operators = operators
        self.name_value_op = name_value_op