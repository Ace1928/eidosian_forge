from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def _span(fg: str, bg: str, string: str) -> str:
    if not s:
        return ''
    return f'<span style="color:{fg};background:{bg}{extra}">{html.escape(string)}</span>'