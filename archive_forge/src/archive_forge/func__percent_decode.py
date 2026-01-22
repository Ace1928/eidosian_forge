import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _percent_decode(text, normalize_case=False, subencoding='utf-8', raise_subencoding_exc=False, encode_stray_percents=False, _decode_map=_HEX_CHAR_MAP):
    """Convert percent-encoded text characters to their normal,
    human-readable equivalents.

    All characters in the input text must be encodable by
    *subencoding*. All special characters underlying the values in the
    percent-encoding must be decodable as *subencoding*. If a
    non-*subencoding*-valid string is passed, the original text is
    returned with no changes applied.

    Only called by field-tailored variants, e.g.,
    :func:`_decode_path_part`, as every percent-encodable part of the
    URL has characters which should not be percent decoded.

    >>> _percent_decode(u'abc%20def')
    u'abc def'

    Args:
        text: Text with percent-encoding present.
        normalize_case: Whether undecoded percent segments, such as encoded
            delimiters, should be uppercased, per RFC 3986 Section 2.1.
            See :func:`_decode_path_part` for an example.
        subencoding: The name of the encoding underlying the percent-encoding.
        raise_subencoding_exc: Whether an error in decoding the bytes
            underlying the percent-decoding should be raised.

    Returns:
        Text: The percent-decoded version of *text*, decoded by *subencoding*.
    """
    try:
        quoted_bytes = text.encode(subencoding)
    except UnicodeEncodeError:
        return text
    bits = quoted_bytes.split(b'%')
    if len(bits) == 1:
        return text
    res = [bits[0]]
    append = res.append
    for item in bits[1:]:
        hexpair, rest = (item[:2], item[2:])
        try:
            append(_decode_map[hexpair])
            append(rest)
        except KeyError:
            pair_is_hex = hexpair in _HEX_CHAR_MAP
            if pair_is_hex or not encode_stray_percents:
                append(b'%')
            else:
                append(b'%25')
            if normalize_case and pair_is_hex:
                append(hexpair.upper())
                append(rest)
            else:
                append(item)
    unquoted_bytes = b''.join(res)
    try:
        return unquoted_bytes.decode(subencoding)
    except UnicodeDecodeError:
        if raise_subencoding_exc:
            raise
        return text