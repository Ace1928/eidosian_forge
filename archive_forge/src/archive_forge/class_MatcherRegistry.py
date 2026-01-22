from __future__ import annotations
from fnmatch import fnmatch
from re import match as rematch
from typing import Callable, cast
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str
class MatcherRegistry:
    """Pattern matching function registry."""
    MatcherNotInstalled = MatcherNotInstalled
    matcher_pattern_first = ['pcre']

    def __init__(self) -> None:
        self._matchers: dict[str, MatcherFunction] = {}
        self._default_matcher: MatcherFunction | None = None

    def register(self, name: str, matcher: MatcherFunction) -> None:
        """Add matcher by name to the registry."""
        self._matchers[name] = matcher

    def unregister(self, name: str) -> None:
        """Remove matcher by name from the registry."""
        try:
            self._matchers.pop(name)
        except KeyError:
            raise self.MatcherNotInstalled(f'No matcher installed for {name}')

    def _set_default_matcher(self, name: str) -> None:
        """Set the default matching method.

        :param name: The name of the registered matching method.
            For example, `glob` (default), `pcre`, or any custom
            methods registered using :meth:`register`.

        :raises MatcherNotInstalled: If the matching method requested
            is not available.
        """
        try:
            self._default_matcher = self._matchers[name]
        except KeyError:
            raise self.MatcherNotInstalled(f'No matcher installed for {name}')

    def match(self, data: bytes, pattern: bytes, matcher: str | None=None, matcher_kwargs: dict[str, str] | None=None) -> bool:
        """Call the matcher."""
        if matcher and (not self._matchers.get(matcher)):
            raise self.MatcherNotInstalled(f'No matcher installed for {matcher}')
        match_func = self._matchers[matcher or 'glob']
        if matcher in self.matcher_pattern_first:
            first_arg = bytes_to_str(pattern)
            second_arg = bytes_to_str(data)
        else:
            first_arg = bytes_to_str(data)
            second_arg = bytes_to_str(pattern)
        return match_func(first_arg, second_arg, **matcher_kwargs or {})