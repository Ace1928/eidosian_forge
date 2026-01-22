from __future__ import annotations
import re
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
from operator import attrgetter
from marshmallow import types
from marshmallow.exceptions import ValidationError
class Regexp(Validator):
    """Validator which succeeds if the ``value`` matches ``regex``.

    .. note::

        Uses `re.match`, which searches for a match at the beginning of a string.

    :param regex: The regular expression string to use. Can also be a compiled
        regular expression pattern.
    :param flags: The regexp flags to use, for example re.IGNORECASE. Ignored
        if ``regex`` is not a string.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}` and `{regex}`.
    """
    default_message = 'String does not match expected pattern.'

    def __init__(self, regex: str | bytes | typing.Pattern, flags: int=0, *, error: str | None=None):
        self.regex = re.compile(regex, flags) if isinstance(regex, (str, bytes)) else regex
        self.error = error or self.default_message

    def _repr_args(self) -> str:
        return f'regex={self.regex!r}'

    def _format_error(self, value: str | bytes) -> str:
        return self.error.format(input=value, regex=self.regex.pattern)

    @typing.overload
    def __call__(self, value: str) -> str:
        ...

    @typing.overload
    def __call__(self, value: bytes) -> bytes:
        ...

    def __call__(self, value):
        if self.regex.match(value) is None:
            raise ValidationError(self._format_error(value))
        return value