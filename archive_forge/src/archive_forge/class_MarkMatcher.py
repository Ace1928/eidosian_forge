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
class MarkMatcher:
    """A matcher for markers which are present.

    Tries to match on any marker names, attached to the given colitem.
    """
    __slots__ = ('own_mark_names',)
    own_mark_names: AbstractSet[str]

    @classmethod
    def from_item(cls, item: 'Item') -> 'MarkMatcher':
        mark_names = {mark.name for mark in item.iter_markers()}
        return cls(mark_names)

    def __call__(self, name: str) -> bool:
        return name in self.own_mark_names