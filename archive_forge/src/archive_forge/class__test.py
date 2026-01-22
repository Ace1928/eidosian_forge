from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
class _test:

    def __init__(self):
        self.ui = Screen()
        self.l = sorted(_curses_colours)
        for c in self.l:
            self.ui.register_palette([(f'{c} on black', c, 'black', 'underline'), (f'{c} on dark blue', c, 'dark blue', 'bold'), (f'{c} on light gray', c, 'light gray', 'standout')])
        with self.ui.start():
            self.run()

    def run(self) -> None:

        class FakeRender:
            pass
        r = FakeRender()
        text = [f'  has_color = {self.ui.has_color!r}', '']
        attr = [[], []]
        r.coords = {}
        r.cursor = None
        for c in self.l:
            t = ''
            a = []
            for p in (f'{c} on black', f'{c} on dark blue', f'{c} on light gray'):
                a.append((p, 27))
                t += (p + 27 * ' ')[:27]
            text.append(t)
            attr.append(a)
        text += ['', 'return values from get_input(): (q exits)', '']
        attr += [[], [], []]
        cols, rows = self.ui.get_cols_rows()
        keys = None
        while keys != ['q']:
            r.text = ([t.ljust(cols) for t in text] + [''] * rows)[:rows]
            r.attr = (attr + [[] for _ in range(rows)])[:rows]
            self.ui.draw_screen((cols, rows), r)
            keys, raw = self.ui.get_input(raw_keys=True)
            if 'window resize' in keys:
                cols, rows = self.ui.get_cols_rows()
            if not keys:
                continue
            t = ''
            a = []
            for k in keys:
                if isinstance(k, str):
                    k = k.encode(util.get_encoding())
                t += f"'{k}' "
                a += [(None, 1), ('yellow on dark blue', len(k)), (None, 2)]
            text.append(f'{t}: {raw!r}')
            attr.append(a)
            text = text[-rows:]
            attr = attr[-rows:]