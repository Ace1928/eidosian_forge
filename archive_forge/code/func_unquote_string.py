from __future__ import annotations
import re
from collections.abc import Generator
from typing import NamedTuple
def unquote_string(string: str) -> str:
    """Unquote a string with JavaScript rules.  The string has to start with
    string delimiters (``'``, ``"`` or the back-tick/grave accent (for template strings).)
    """
    assert string and string[0] == string[-1] and (string[0] in '"\'`'), 'string provided is not properly delimited'
    string = line_join_re.sub('\\1', string[1:-1])
    result: list[str] = []
    add = result.append
    pos = 0
    while True:
        escape_pos = string.find('\\', pos)
        if escape_pos < 0:
            break
        add(string[pos:escape_pos])
        next_char = string[escape_pos + 1]
        if next_char in escapes:
            add(escapes[next_char])
        elif next_char in 'uU':
            escaped = uni_escape_re.match(string, escape_pos + 2)
            if escaped is not None:
                escaped_value = escaped.group()
                if len(escaped_value) == 4:
                    try:
                        add(chr(int(escaped_value, 16)))
                    except ValueError:
                        pass
                    else:
                        pos = escape_pos + 6
                        continue
                add(next_char + escaped_value)
                pos = escaped.end()
                continue
            else:
                add(next_char)
        elif next_char in 'xX':
            escaped = hex_escape_re.match(string, escape_pos + 2)
            if escaped is not None:
                escaped_value = escaped.group()
                add(chr(int(escaped_value, 16)))
                pos = escape_pos + 2 + len(escaped_value)
                continue
            else:
                add(next_char)
        else:
            add(next_char)
        pos = escape_pos + 2
    if pos < len(string):
        add(string[pos:])
    return ''.join(result)