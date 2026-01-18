from __future__ import annotations
import typing as t
import sqlglot.expressions as exp
from sqlglot.errors import ParseError
from sqlglot.tokens import Token, Tokenizer, TokenType
Takes in a JSON path string and parses it into a JSONPath expression.