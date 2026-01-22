from __future__ import annotations
import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType
from markupsafe import escape
from .repr import debug_repr
from .repr import dump
from .repr import helper
class _InteractiveConsole(code.InteractiveInterpreter):
    locals: dict[str, t.Any]

    def __init__(self, globals: dict[str, t.Any], locals: dict[str, t.Any]) -> None:
        self.loader = _ConsoleLoader()
        locals = {**globals, **locals, 'dump': dump, 'help': helper, '__loader__': self.loader}
        super().__init__(locals)
        original_compile = self.compile

        def compile(source: str, filename: str, symbol: str) -> CodeType | None:
            code = original_compile(source, filename, symbol)
            if code is not None:
                self.loader.register(code, source)
            return code
        self.compile = compile
        self.more = False
        self.buffer: list[str] = []

    def runsource(self, source: str, **kwargs: t.Any) -> str:
        source = f'{source.rstrip()}\n'
        ThreadedStream.push()
        prompt = '... ' if self.more else '>>> '
        try:
            source_to_eval = ''.join(self.buffer + [source])
            if super().runsource(source_to_eval, '<debugger>', 'single'):
                self.more = True
                self.buffer.append(source)
            else:
                self.more = False
                del self.buffer[:]
        finally:
            output = ThreadedStream.fetch()
        return f'{prompt}{escape(source)}{output}'

    def runcode(self, code: CodeType) -> None:
        try:
            exec(code, self.locals)
        except Exception:
            self.showtraceback()

    def showtraceback(self) -> None:
        from .tbtools import DebugTraceback
        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=1)
        sys.stdout._write(te.render_traceback_html())

    def showsyntaxerror(self, filename: str | None=None) -> None:
        from .tbtools import DebugTraceback
        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=4)
        sys.stdout._write(te.render_traceback_html())

    def write(self, data: str) -> None:
        sys.stdout.write(data)