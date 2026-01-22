from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class NamespaceParser(Parser, metaclass=abc.ABCMeta):
    """Base class for composite argument parsers that store their results in a namespace."""

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        namespace = state.current_namespace
        current = getattr(namespace, self.dest)
        if current and self.limit_one:
            if state.mode == ParserMode.PARSE:
                raise ParserError('Option cannot be specified more than once.')
            raise CompletionError('Option cannot be specified more than once.')
        value = self.get_value(state)
        if self.use_list:
            if not current:
                current = []
                setattr(namespace, self.dest, current)
            current.append(value)
        else:
            setattr(namespace, self.dest, value)
        return value

    def get_value(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result, without storing the result in the namespace."""
        return super().parse(state)

    @property
    def use_list(self) -> bool:
        """True if the destination is a list, otherwise False."""
        return False

    @property
    def limit_one(self) -> bool:
        """True if only one target is allowed, otherwise False."""
        return not self.use_list

    @property
    @abc.abstractmethod
    def dest(self) -> str:
        """The name of the attribute where the value should be stored."""