import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@staticmethod
def _split_gpg_and_payload(sequence, strict=None):
    if not strict:
        strict = {}
    gpg_pre_lines = []
    lines = []
    gpg_post_lines = []
    state = b'SAFE'
    accept_empty_or_whitespace = strict.get('whitespace-separates-paragraphs', True)
    first_line = True
    for line in sequence:
        line = line.strip(b'\r\n')
        if first_line:
            if not line or line.isspace():
                continue
            first_line = False
        m = Deb822._gpgre.match(line) if line.startswith(b'-') else None
        is_empty_line = not line or line.isspace() if accept_empty_or_whitespace else not line
        if not m:
            if state == b'SAFE':
                if not is_empty_line:
                    lines.append(line)
                elif not gpg_pre_lines:
                    break
            elif state == b'SIGNED MESSAGE':
                if is_empty_line:
                    state = b'SAFE'
                else:
                    gpg_pre_lines.append(line)
            elif state == b'SIGNATURE':
                gpg_post_lines.append(line)
        else:
            if m.group('action') == b'BEGIN':
                state = m.group('what')
            elif m.group('action') == b'END':
                gpg_post_lines.append(line)
                break
            if not is_empty_line:
                if not lines:
                    gpg_pre_lines.append(line)
                else:
                    gpg_post_lines.append(line)
    if lines:
        return (gpg_pre_lines, lines, gpg_post_lines)
    raise EOFError('only blank lines found in input')