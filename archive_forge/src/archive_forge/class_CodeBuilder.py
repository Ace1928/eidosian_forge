from __future__ import annotations
import re
from typing import (
class CodeBuilder:
    """Build source code conveniently."""

    def __init__(self, indent: int=0) -> None:
        self.code: list[str | CodeBuilder] = []
        self.indent_level = indent

    def __str__(self) -> str:
        return ''.join((str(c) for c in self.code))

    def add_line(self, line: str) -> None:
        """Add a line of source to the code.

        Indentation and newline will be added for you, don't provide them.

        """
        self.code.extend([' ' * self.indent_level, line, '\n'])

    def add_section(self) -> CodeBuilder:
        """Add a section, a sub-CodeBuilder."""
        section = CodeBuilder(self.indent_level)
        self.code.append(section)
        return section
    INDENT_STEP = 4

    def indent(self) -> None:
        """Increase the current indent for following lines."""
        self.indent_level += self.INDENT_STEP

    def dedent(self) -> None:
        """Decrease the current indent for following lines."""
        self.indent_level -= self.INDENT_STEP

    def get_globals(self) -> dict[str, Any]:
        """Execute the code, and return a dict of globals it defines."""
        assert self.indent_level == 0
        python_source = str(self)
        global_namespace: dict[str, Any] = {}
        exec(python_source, global_namespace)
        return global_namespace