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
def _standard_operators(self):
    return [('.', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('?.', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('[]', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('{}', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('+', OperatorType.PREFIX_UNARY), ('-', OperatorType.PREFIX_UNARY), (), ('=~', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('!~', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('*', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('/', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('mod', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('+', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('-', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('>', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('<', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('>=', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('<=', OperatorType.BINARY_LEFT_ASSOCIATIVE), ('!=', OperatorType.BINARY_LEFT_ASSOCIATIVE, 'not_equal'), ('=', OperatorType.BINARY_LEFT_ASSOCIATIVE, 'equal'), ('in', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('not', OperatorType.PREFIX_UNARY), (), ('and', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('or', OperatorType.BINARY_LEFT_ASSOCIATIVE), (), ('->', OperatorType.BINARY_RIGHT_ASSOCIATIVE)]