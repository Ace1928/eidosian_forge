from __future__ import annotations
import abc
import argparse
import os
import re
import typing as t
from .argcompletion import (
from .parsers import (
class CompositeActionCompletionFinder(RegisteredCompletionFinder):
    """Completion finder with support for composite argument parsing."""

    def get_completions(self, prefix: str, action: argparse.Action, parsed_args: argparse.Namespace) -> list[str]:
        """Return a list of completions appropriate for the given prefix and action, taking into account the arguments that have already been parsed."""
        assert isinstance(action, CompositeAction)
        state = ParserState(mode=ParserMode.LIST if self.list_mode else ParserMode.COMPLETE, remainder=prefix, namespaces=[parsed_args])
        answer = complete(action.definition, state)
        completions = []
        if isinstance(answer, CompletionSuccess):
            self.disable_completion_mangling = answer.preserve
            completions = answer.completions
        if isinstance(answer, CompletionError):
            warn(answer.message)
        return completions