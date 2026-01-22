from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def apply_text_layout(text: str | bytes, attr: list[tuple[Hashable, int]], ls: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]], maxcol: int) -> TextCanvas:
    t: list[bytes] = []
    a: list[list[tuple[Hashable | None, int]]] = []
    c: list[list[tuple[Literal['0', 'U'] | None, int]]] = []
    aw = _AttrWalk()

    def arange(start_offs: int, end_offs: int) -> list[tuple[Hashable | None, int]]:
        """Return an attribute list for the range of text specified."""
        if start_offs < aw.offset:
            aw.counter = 0
            aw.offset = 0
        o = []
        while aw.offset <= end_offs:
            if len(attr) <= aw.counter:
                o.append((None, end_offs - max(start_offs, aw.offset)))
                break
            at, run = attr[aw.counter]
            if aw.offset + run <= start_offs:
                aw.counter += 1
                aw.offset += run
                continue
            if end_offs <= aw.offset + run:
                o.append((at, end_offs - max(start_offs, aw.offset)))
                break
            o.append((at, aw.offset + run - max(start_offs, aw.offset)))
            aw.counter += 1
            aw.offset += run
        return o
    for line_layout in ls:
        line_layout = trim_line(line_layout, text, 0, maxcol)
        line = []
        linea = []
        linec = []

        def attrrange(start_offs: int, end_offs: int, destw: int) -> None:
            """
            Add attributes based on attributes between
            start_offs and end_offs.
            """
            if start_offs == end_offs:
                [(at, run)] = arange(start_offs, end_offs)
                rle_append_modify(linea, (at, destw))
                return
            if destw == end_offs - start_offs:
                for at, run in arange(start_offs, end_offs):
                    rle_append_modify(linea, (at, run))
                return
            o = start_offs
            for at, run in arange(start_offs, end_offs):
                if o + run == end_offs:
                    rle_append_modify(linea, (at, destw))
                    return
                tseg = text[o:o + run]
                tseg, cs = apply_target_encoding(tseg)
                segw = rle_len(cs)
                rle_append_modify(linea, (at, segw))
                o += run
                destw -= segw
        for seg in line_layout:
            s = LayoutSegment(seg)
            if s.end:
                tseg, cs = apply_target_encoding(text[s.offs:s.end])
                line.append(tseg)
                attrrange(s.offs, s.end, rle_len(cs))
                rle_join_modify(linec, cs)
            elif s.text:
                tseg, cs = apply_target_encoding(s.text)
                line.append(tseg)
                attrrange(s.offs, s.offs, len(tseg))
                rle_join_modify(linec, cs)
            elif s.offs:
                if s.sc:
                    line.append(b''.rjust(s.sc))
                    attrrange(s.offs, s.offs, s.sc)
            else:
                line.append(b''.rjust(s.sc))
                linea.append((None, s.sc))
                linec.append((None, s.sc))
        t.append(b''.join(line))
        a.append(linea)
        c.append(linec)
    return TextCanvas(t, a, c, maxcol=maxcol)