from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
class PaginatedListBase(Generic[T]):
    __elements: List[T]

    def _couldGrow(self) -> bool:
        raise NotImplementedError

    def _fetchNextPage(self) -> List[T]:
        raise NotImplementedError

    def __init__(self, elements: Optional[List[T]]=None) -> None:
        self.__elements = [] if elements is None else elements

    def __getitem__(self, index: Union[int, slice]) -> Any:
        assert isinstance(index, (int, slice))
        if isinstance(index, int):
            self.__fetchToIndex(index)
            return self.__elements[index]
        else:
            return self._Slice(self, index)

    def __iter__(self) -> Iterator[T]:
        yield from self.__elements
        while self._couldGrow():
            newElements = self._grow()
            yield from newElements

    def _isBiggerThan(self, index: int) -> bool:
        return len(self.__elements) > index or self._couldGrow()

    def __fetchToIndex(self, index: int) -> None:
        while len(self.__elements) <= index and self._couldGrow():
            self._grow()

    def _grow(self) -> List[T]:
        newElements = self._fetchNextPage()
        self.__elements += newElements
        return newElements

    class _Slice:

        def __init__(self, theList: 'PaginatedListBase[T]', theSlice: slice):
            self.__list = theList
            self.__start = theSlice.start or 0
            self.__stop = theSlice.stop
            self.__step = theSlice.step or 1

        def __iter__(self) -> Iterator[T]:
            index = self.__start
            while not self.__finished(index):
                if self.__list._isBiggerThan(index):
                    yield self.__list[index]
                    index += self.__step
                else:
                    return

        def __finished(self, index: int) -> bool:
            return self.__stop is not None and index >= self.__stop