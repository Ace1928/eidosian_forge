import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def visit_STRING(self, leaf: Leaf) -> Iterator[Line]:
    if Preview.hex_codes_in_unicode_sequences in self.mode:
        normalize_unicode_escape_sequences(leaf)
    if is_docstring(leaf, self.mode) and (not re.search('\\\\\\s*\\n', leaf.value)):
        if self.mode.string_normalization:
            docstring = normalize_string_prefix(leaf.value)
            docstring = normalize_string_quotes(docstring)
        else:
            docstring = leaf.value
        prefix = get_string_prefix(docstring)
        docstring = docstring[len(prefix):]
        quote_char = docstring[0]
        quote_len = 1 if docstring[1] != quote_char else 3
        docstring = docstring[quote_len:-quote_len]
        docstring_started_empty = not docstring
        indent = ' ' * 4 * self.current_line.depth
        if is_multiline_string(leaf):
            docstring = fix_docstring(docstring, indent)
        else:
            docstring = docstring.strip()
        has_trailing_backslash = False
        if docstring:
            if docstring[0] == quote_char:
                docstring = ' ' + docstring
            if docstring[-1] == quote_char:
                docstring += ' '
            if docstring[-1] == '\\':
                backslash_count = len(docstring) - len(docstring.rstrip('\\'))
                if backslash_count % 2:
                    docstring += ' '
                    has_trailing_backslash = True
        elif not docstring_started_empty:
            docstring = ' '
        quote = quote_char * quote_len
        if quote_len == 3:
            lines = docstring.splitlines()
            last_line_length = len(lines[-1]) if docstring else 0
            if len(lines) > 1 and last_line_length + quote_len > self.mode.line_length and (len(indent) + quote_len <= self.mode.line_length) and (not has_trailing_backslash):
                if Preview.docstring_check_for_newline in self.mode and leaf.value[-1 - quote_len] == '\n':
                    leaf.value = prefix + quote + docstring + quote
                else:
                    leaf.value = prefix + quote + docstring + '\n' + indent + quote
            else:
                leaf.value = prefix + quote + docstring + quote
        else:
            leaf.value = prefix + quote + docstring + quote
    yield from self.visit_default(leaf)