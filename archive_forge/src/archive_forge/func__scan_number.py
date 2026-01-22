from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_number(self) -> None:
    if self._char == '0':
        peek = self._peek.upper()
        if peek == 'B':
            return self._scan_bits() if self.BIT_STRINGS else self._add(TokenType.NUMBER)
        elif peek == 'X':
            return self._scan_hex() if self.HEX_STRINGS else self._add(TokenType.NUMBER)
    decimal = False
    scientific = 0
    while True:
        if self._peek.isdigit():
            self._advance()
        elif self._peek == '.' and (not decimal):
            decimal = True
            self._advance()
        elif self._peek in ('-', '+') and scientific == 1:
            scientific += 1
            self._advance()
        elif self._peek.upper() == 'E' and (not scientific):
            scientific += 1
            self._advance()
        elif self._peek.isidentifier():
            number_text = self._text
            literal = ''
            while self._peek.strip() and self._peek not in self.SINGLE_TOKENS:
                literal += self._peek
                self._advance()
            token_type = self.KEYWORDS.get(self.NUMERIC_LITERALS.get(literal.upper(), ''))
            if token_type:
                self._add(TokenType.NUMBER, number_text)
                self._add(TokenType.DCOLON, '::')
                return self._add(token_type, literal)
            elif self.dialect.IDENTIFIERS_CAN_START_WITH_DIGIT:
                return self._add(TokenType.VAR)
            self._advance(-len(literal))
            return self._add(TokenType.NUMBER, number_text)
        else:
            return self._add(TokenType.NUMBER)