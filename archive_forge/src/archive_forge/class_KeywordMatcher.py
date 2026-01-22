import dataclasses
from typing import AbstractSet
from typing import Collection
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from .expression import Expression
from .expression import ParseError
from .structures import EMPTY_PARAMETERSET_OPTION
from .structures import get_empty_parameterset_mark
from .structures import Mark
from .structures import MARK_GEN
from .structures import MarkDecorator
from .structures import MarkGenerator
from .structures import ParameterSet
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
@dataclasses.dataclass
class KeywordMatcher:
    """A matcher for keywords.

    Given a list of names, matches any substring of one of these names. The
    string inclusion check is case-insensitive.

    Will match on the name of colitem, including the names of its parents.
    Only matches names of items which are either a :class:`Class` or a
    :class:`Function`.

    Additionally, matches on names in the 'extra_keyword_matches' set of
    any item, as well as names directly assigned to test functions.
    """
    __slots__ = ('_names',)
    _names: AbstractSet[str]

    @classmethod
    def from_item(cls, item: 'Item') -> 'KeywordMatcher':
        mapped_names = set()
        import pytest
        for node in item.listchain():
            if isinstance(node, pytest.Session):
                continue
            if isinstance(node, pytest.Directory) and isinstance(node.parent, pytest.Session):
                continue
            mapped_names.add(node.name)
        mapped_names.update(item.listextrakeywords())
        function_obj = getattr(item, 'function', None)
        if function_obj:
            mapped_names.update(function_obj.__dict__)
        mapped_names.update((mark.name for mark in item.iter_markers()))
        return cls(mapped_names)

    def __call__(self, subname: str) -> bool:
        subname = subname.lower()
        names = (name.lower() for name in self._names)
        for name in names:
            if subname in name:
                return True
        return False