import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple

    items : items NUM_VAL
          | items WORD
          | items STRING
          | items QUOTEDSTRING
          | items COMMA
          | items COLON
          | items LBRACE
          | items RBRACE
          | items LBRACKET
          | items RBRACKET
          | items TR
          | items LPAREN
          | items RPAREN
          | items ASTERISK
          | items EQ
          | items SET
          | items TABLE
          | items PARAM
          | NUM_VAL
          | WORD
          | STRING
          | QUOTEDSTRING
          | COMMA
          | COLON
          | LBRACKET
          | RBRACKET
          | LBRACE
          | RBRACE
          | TR
          | LPAREN
          | RPAREN
          | ASTERISK
          | EQ
          | SET
          | TABLE
          | PARAM
    