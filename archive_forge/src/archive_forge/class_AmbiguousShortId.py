from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
class AmbiguousShortId(Exception):
    """The short id is ambiguous."""

    def __init__(self, prefix, options) -> None:
        self.prefix = prefix
        self.options = options