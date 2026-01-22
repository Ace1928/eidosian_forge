from __future__ import annotations
from fnmatch import fnmatch
from re import match as rematch
from typing import Callable, cast
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str
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