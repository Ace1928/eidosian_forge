from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class ChoicesParser(DynamicChoicesParser):
    """Composite argument parser which relies on a static list of choices."""

    def __init__(self, choices: list[str], conditions: MatchConditions=MatchConditions.CHOICE) -> None:
        self.choices = choices
        super().__init__(conditions=conditions)

    def get_choices(self, value: str) -> list[str]:
        """Return a list of valid choices based on the given input value."""
        return self.choices

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        return '|'.join(self.choices)