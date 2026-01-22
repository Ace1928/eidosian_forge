from typing import Any, Dict, Iterator, List, Protocol, TypeVar, Union
class SimplePath(Protocol):
    """
    A minimal subset of pathlib.Path required by PathDistribution.
    """

    def joinpath(self) -> 'SimplePath':
        ...

    def __truediv__(self) -> 'SimplePath':
        ...

    def parent(self) -> 'SimplePath':
        ...

    def read_text(self) -> str:
        ...