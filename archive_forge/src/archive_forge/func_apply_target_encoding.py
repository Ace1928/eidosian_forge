from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def apply_target_encoding(s: str | bytes) -> tuple[bytes, list[tuple[Literal['U', '0'] | None, int]]]:
    """
    Return (encoded byte string, character set rle).
    """
    from urwid.display import escape
    if _use_dec_special and isinstance(s, str):
        s = s.translate(escape.DEC_SPECIAL_CHARMAP)
    if isinstance(s, str):
        s = s.replace(escape.SI + escape.SO, '')
        s = codecs.encode(s, _target_encoding, 'replace')
    if not isinstance(s, bytes):
        raise TypeError(s)
    SO = escape.SO.encode('ascii')
    SI = escape.SI.encode('ascii')
    sis = s.split(SO)
    sis0 = sis[0].replace(SI, b'')
    sout = []
    cout = []
    if sis0:
        sout.append(sis0)
        cout.append((None, len(sis0)))
    if len(sis) == 1:
        return (sis0, cout)
    for sn in sis[1:]:
        sl = sn.split(SI, 1)
        if len(sl) == 1:
            sin = sl[0]
            sout.append(sin)
            rle_append_modify(cout, (escape.DEC_TAG, len(sin)))
            continue
        sin, son = sl
        son = son.replace(SI, b'')
        if sin:
            sout.append(sin)
            rle_append_modify(cout, (escape.DEC_TAG, len(sin)))
        if son:
            sout.append(son)
            rle_append_modify(cout, (None, len(son)))
    outstr = b''.join(sout)
    return (outstr, cout)