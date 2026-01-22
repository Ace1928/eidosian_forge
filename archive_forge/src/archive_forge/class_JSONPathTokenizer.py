from __future__ import annotations
import typing as t
import sqlglot.expressions as exp
from sqlglot.errors import ParseError
from sqlglot.tokens import Token, Tokenizer, TokenType
class JSONPathTokenizer(Tokenizer):
    SINGLE_TOKENS = {'(': TokenType.L_PAREN, ')': TokenType.R_PAREN, '[': TokenType.L_BRACKET, ']': TokenType.R_BRACKET, ':': TokenType.COLON, ',': TokenType.COMMA, '-': TokenType.DASH, '.': TokenType.DOT, '?': TokenType.PLACEHOLDER, '@': TokenType.PARAMETER, "'": TokenType.QUOTE, '"': TokenType.QUOTE, '$': TokenType.DOLLAR, '*': TokenType.STAR}
    KEYWORDS = {'..': TokenType.DOT}
    IDENTIFIER_ESCAPES = ['\\']
    STRING_ESCAPES = ['\\']