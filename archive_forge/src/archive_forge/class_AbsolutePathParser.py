from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class AbsolutePathParser(Parser):
    """Composite argument parser for absolute file paths. Paths are only verified for proper syntax, not for existence."""

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        path = ''
        with state.delimit(PATH_DELIMITER, required=False) as boundary:
            while boundary.ready:
                if path:
                    path += AnyParser(nothing=True).parse(state)
                else:
                    path += ChoicesParser([PATH_DELIMITER]).parse(state)
                path += boundary.match or ''
        return path