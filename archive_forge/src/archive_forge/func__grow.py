from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def _grow(self) -> List[T]:
    newElements = self._fetchNextPage()
    self.__elements += newElements
    return newElements