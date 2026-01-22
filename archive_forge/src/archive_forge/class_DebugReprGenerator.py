from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
class DebugReprGenerator:

    def __init__(self) -> None:
        self._stack: list[t.Any] = []
    list_repr = _sequence_repr_maker('[', ']', list)
    tuple_repr = _sequence_repr_maker('(', ')', tuple)
    set_repr = _sequence_repr_maker('set([', '])', set)
    frozenset_repr = _sequence_repr_maker('frozenset([', '])', frozenset)
    deque_repr = _sequence_repr_maker('<span class="module">collections.</span>deque([', '])', deque)

    def regex_repr(self, obj: t.Pattern) -> str:
        pattern = repr(obj.pattern)
        pattern = codecs.decode(pattern, 'unicode-escape', 'ignore')
        pattern = f'r{pattern}'
        return f're.compile(<span class="string regex">{pattern}</span>)'

    def string_repr(self, obj: str | bytes, limit: int=70) -> str:
        buf = ['<span class="string">']
        r = repr(obj)
        if len(r) - limit > 2:
            buf.extend((escape(r[:limit]), '<span class="extended">', escape(r[limit:]), '</span>'))
        else:
            buf.append(escape(r))
        buf.append('</span>')
        out = ''.join(buf)
        if r[0] in '\'"' or (r[0] == 'b' and r[1] in '\'"'):
            return _add_subclass_info(out, obj, (bytes, str))
        return out

    def dict_repr(self, d: dict[int, None] | dict[str, int] | dict[str | int, int], recursive: bool, limit: int=5) -> str:
        if recursive:
            return _add_subclass_info('{...}', d, dict)
        buf = ['{']
        have_extended_section = False
        for idx, (key, value) in enumerate(d.items()):
            if idx:
                buf.append(', ')
            if idx == limit - 1:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(f'<span class="pair"><span class="key">{self.repr(key)}</span>: <span class="value">{self.repr(value)}</span></span>')
        if have_extended_section:
            buf.append('</span>')
        buf.append('}')
        return _add_subclass_info(''.join(buf), d, dict)

    def object_repr(self, obj: type[dict] | t.Callable | type[list] | None) -> str:
        r = repr(obj)
        return f'<span class="object">{escape(r)}</span>'

    def dispatch_repr(self, obj: t.Any, recursive: bool) -> str:
        if obj is helper:
            return f'<span class="help">{helper!r}</span>'
        if isinstance(obj, (int, float, complex)):
            return f'<span class="number">{obj!r}</span>'
        if isinstance(obj, str) or isinstance(obj, bytes):
            return self.string_repr(obj)
        if isinstance(obj, RegexType):
            return self.regex_repr(obj)
        if isinstance(obj, list):
            return self.list_repr(obj, recursive)
        if isinstance(obj, tuple):
            return self.tuple_repr(obj, recursive)
        if isinstance(obj, set):
            return self.set_repr(obj, recursive)
        if isinstance(obj, frozenset):
            return self.frozenset_repr(obj, recursive)
        if isinstance(obj, dict):
            return self.dict_repr(obj, recursive)
        if isinstance(obj, deque):
            return self.deque_repr(obj, recursive)
        return self.object_repr(obj)

    def fallback_repr(self) -> str:
        try:
            info = ''.join(format_exception_only(*sys.exc_info()[:2]))
        except Exception:
            info = '?'
        return f'<span class="brokenrepr">&lt;broken repr ({escape(info.strip())})&gt;</span>'

    def repr(self, obj: object) -> str:
        recursive = False
        for item in self._stack:
            if item is obj:
                recursive = True
                break
        self._stack.append(obj)
        try:
            try:
                return self.dispatch_repr(obj, recursive)
            except Exception:
                return self.fallback_repr()
        finally:
            self._stack.pop()

    def dump_object(self, obj: object) -> str:
        repr = None
        items: list[tuple[str, str]] | None = None
        if isinstance(obj, dict):
            title = 'Contents of'
            items = []
            for key, value in obj.items():
                if not isinstance(key, str):
                    items = None
                    break
                items.append((key, self.repr(value)))
        if items is None:
            items = []
            repr = self.repr(obj)
            for key in dir(obj):
                try:
                    items.append((key, self.repr(getattr(obj, key))))
                except Exception:
                    pass
            title = 'Details for'
        title += f' {object.__repr__(obj)[1:-1]}'
        return self.render_object_dump(items, title, repr)

    def dump_locals(self, d: dict[str, t.Any]) -> str:
        items = [(key, self.repr(value)) for key, value in d.items()]
        return self.render_object_dump(items, 'Local variables in frame')

    def render_object_dump(self, items: list[tuple[str, str]], title: str, repr: str | None=None) -> str:
        html_items = []
        for key, value in items:
            html_items.append(f'<tr><th>{escape(key)}<td><pre class=repr>{value}</pre>')
        if not html_items:
            html_items.append('<tr><td><em>Nothing</em>')
        return OBJECT_DUMP_HTML % {'title': escape(title), 'repr': f'<pre class=repr>{(repr if repr else '')}</pre>', 'items': '\n'.join(html_items)}