from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def __fetchToIndex(self, index: int) -> None:
    while len(self.__elements) <= index and self._couldGrow():
        self._grow()