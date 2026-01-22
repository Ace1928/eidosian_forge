from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def _find_fstring_string(endpats, fstring_stack, line, lnum, pos):
    tos = fstring_stack[-1]
    allow_multiline = tos.allow_multiline()
    if tos.is_in_format_spec():
        if allow_multiline:
            regex = fstring_format_spec_multi_line
        else:
            regex = fstring_format_spec_single_line
    elif allow_multiline:
        regex = fstring_string_multi_line
    else:
        regex = fstring_string_single_line
    match = regex.match(line, pos)
    if match is None:
        return (tos.previous_lines, pos)
    if not tos.previous_lines:
        tos.last_string_start_pos = (lnum, pos)
    string = match.group(0)
    for fstring_stack_node in fstring_stack:
        end_match = endpats[fstring_stack_node.quote].match(string)
        if end_match is not None:
            string = end_match.group(0)[:-len(fstring_stack_node.quote)]
    new_pos = pos
    new_pos += len(string)
    if string.endswith('\n') or string.endswith('\r'):
        tos.previous_lines += string
        string = ''
    else:
        string = tos.previous_lines + string
    return (string, new_pos)