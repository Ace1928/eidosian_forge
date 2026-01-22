import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _parse_section_header_line(line: bytes) -> Tuple[Section, bytes]:
    line = _strip_comments(line).rstrip()
    in_quotes = False
    escaped = False
    for i, c in enumerate(line):
        if escaped:
            escaped = False
            continue
        if c == ord(b'"'):
            in_quotes = not in_quotes
        if c == ord(b'\\'):
            escaped = True
        if c == ord(b']') and (not in_quotes):
            last = i
            break
    else:
        raise ValueError('expected trailing ]')
    pts = line[1:last].split(b' ', 1)
    line = line[last + 1:]
    section: Section
    if len(pts) == 2:
        if pts[1][:1] != b'"' or pts[1][-1:] != b'"':
            raise ValueError('Invalid subsection %r' % pts[1])
        else:
            pts[1] = pts[1][1:-1]
        if not _check_section_name(pts[0]):
            raise ValueError('invalid section name %r' % pts[0])
        section = (pts[0], pts[1])
    else:
        if not _check_section_name(pts[0]):
            raise ValueError('invalid section name %r' % pts[0])
        pts = pts[0].split(b'.', 1)
        if len(pts) == 2:
            section = (pts[0], pts[1])
        else:
            section = (pts[0],)
    return (section, line)