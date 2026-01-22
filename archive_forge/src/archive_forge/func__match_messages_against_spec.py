from __future__ import annotations
import ast
import io
import os
import sys
import tokenize
from collections.abc import (
from os.path import relpath
from textwrap import dedent
from tokenize import COMMENT, NAME, OP, STRING, generate_tokens
from typing import TYPE_CHECKING, Any
from babel.util import parse_encoding, parse_future_flags, pathmatch
def _match_messages_against_spec(lineno: int, messages: list[str | None], comments: list[str], fileobj: _FileObj, spec: tuple[int | tuple[int, str], ...]):
    translatable = []
    context = None
    last_index = len(messages)
    for index in spec:
        if isinstance(index, tuple):
            context = messages[index[0] - 1]
            continue
        if last_index < index:
            return
        message = messages[index - 1]
        if message is None:
            return
        translatable.append(message)
    if isinstance(spec[0], tuple):
        first_msg_index = spec[1] - 1
    else:
        first_msg_index = spec[0] - 1
    if not messages[first_msg_index]:
        filename = getattr(fileobj, 'name', None) or '(unknown)'
        sys.stderr.write(f'{filename}:{lineno}: warning: Empty msgid.  It is reserved by GNU gettext: gettext("") returns the header entry with meta information, not the empty string.\n')
        return
    translatable = tuple(translatable)
    if len(translatable) == 1:
        translatable = translatable[0]
    return (lineno, translatable, comments, context)