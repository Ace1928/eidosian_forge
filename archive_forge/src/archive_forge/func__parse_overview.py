import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _parse_overview(lines, fmt, data_process_func=None):
    """Parse the response to an OVER or XOVER command according to the
    overview format `fmt`."""
    n_defaults = len(_DEFAULT_OVERVIEW_FMT)
    overview = []
    for line in lines:
        fields = {}
        article_number, *tokens = line.split('\t')
        article_number = int(article_number)
        for i, token in enumerate(tokens):
            if i >= len(fmt):
                continue
            field_name = fmt[i]
            is_metadata = field_name.startswith(':')
            if i >= n_defaults and (not is_metadata):
                h = field_name + ': '
                if token and token[:len(h)].lower() != h:
                    raise NNTPDataError("OVER/XOVER response doesn't include names of additional headers")
                token = token[len(h):] if token else None
            fields[fmt[i]] = token
        overview.append((article_number, fields))
    return overview