from __future__ import annotations
import os.path
import types
import zipimport
from typing import Iterable, TYPE_CHECKING
from coverage import env
from coverage.exceptions import CoverageException, NoSource
from coverage.files import canonical_filename, relative_filename, zip_location
from coverage.misc import expensive, isolate_module, join_regex
from coverage.parser import PythonParser
from coverage.phystokens import source_token_lines, source_encoding
from coverage.plugin import FileReporter
from coverage.types import TArc, TLineNo, TMorf, TSourceTokenLines
class PythonFileReporter(FileReporter):
    """Report support for a Python file."""

    def __init__(self, morf: TMorf, coverage: Coverage | None=None) -> None:
        self.coverage = coverage
        filename = source_for_morf(morf)
        fname = filename
        canonicalize = True
        if self.coverage is not None:
            if self.coverage.config.relative_files:
                canonicalize = False
        if canonicalize:
            fname = canonical_filename(filename)
        super().__init__(fname)
        if hasattr(morf, '__name__'):
            name = morf.__name__.replace('.', os.sep)
            if os.path.basename(filename).startswith('__init__.'):
                name += os.sep + '__init__'
            name += '.py'
        else:
            name = relative_filename(filename)
        self.relname = name
        self._source: str | None = None
        self._parser: PythonParser | None = None
        self._excluded = None

    def __repr__(self) -> str:
        return f'<PythonFileReporter {self.filename!r}>'

    def relative_filename(self) -> str:
        return self.relname

    @property
    def parser(self) -> PythonParser:
        """Lazily create a :class:`PythonParser`."""
        assert self.coverage is not None
        if self._parser is None:
            self._parser = PythonParser(filename=self.filename, exclude=self.coverage._exclude_regex('exclude'))
            self._parser.parse_source()
        return self._parser

    def lines(self) -> set[TLineNo]:
        """Return the line numbers of statements in the file."""
        return self.parser.statements

    def excluded_lines(self) -> set[TLineNo]:
        """Return the line numbers of statements in the file."""
        return self.parser.excluded

    def translate_lines(self, lines: Iterable[TLineNo]) -> set[TLineNo]:
        return self.parser.translate_lines(lines)

    def translate_arcs(self, arcs: Iterable[TArc]) -> set[TArc]:
        return self.parser.translate_arcs(arcs)

    @expensive
    def no_branch_lines(self) -> set[TLineNo]:
        assert self.coverage is not None
        no_branch = self.parser.lines_matching(join_regex(self.coverage.config.partial_list), join_regex(self.coverage.config.partial_always_list))
        return no_branch

    @expensive
    def arcs(self) -> set[TArc]:
        return self.parser.arcs()

    @expensive
    def exit_counts(self) -> dict[TLineNo, int]:
        return self.parser.exit_counts()

    def missing_arc_description(self, start: TLineNo, end: TLineNo, executed_arcs: Iterable[TArc] | None=None) -> str:
        return self.parser.missing_arc_description(start, end, executed_arcs)

    def source(self) -> str:
        if self._source is None:
            self._source = get_python_source(self.filename)
        return self._source

    def should_be_python(self) -> bool:
        """Does it seem like this file should contain Python?

        This is used to decide if a file reported as part of the execution of
        a program was really likely to have contained Python in the first
        place.

        """
        _, ext = os.path.splitext(self.filename)
        if ext.startswith('.py'):
            return True
        if not ext:
            return True
        return False

    def source_token_lines(self) -> TSourceTokenLines:
        return source_token_lines(self.source())