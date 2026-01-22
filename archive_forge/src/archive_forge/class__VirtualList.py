import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
class _VirtualList(Sequence[PageObject]):

    def __init__(self, length_function: Callable[[], int], get_function: Callable[[int], PageObject]) -> None:
        self.length_function = length_function
        self.get_function = get_function
        self.current = -1

    def __len__(self) -> int:
        return self.length_function()

    @overload
    def __getitem__(self, index: int) -> PageObject:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PageObject]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[PageObject, Sequence[PageObject]]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            cls = type(self)
            return cls(indices.__len__, lambda idx: self[indices[idx]])
        if not isinstance(index, int):
            raise TypeError('sequence indices must be integers')
        len_self = len(self)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('sequence index out of range')
        return self.get_function(index)

    def __delitem__(self, index: Union[int, slice]) -> None:
        if isinstance(index, slice):
            r = list(range(*index.indices(len(self))))
            r.sort()
            r.reverse()
            for p in r:
                del self[p]
            return
        if not isinstance(index, int):
            raise TypeError('index must be integers')
        len_self = len(self)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('index out of range')
        ind = self[index].indirect_reference
        assert ind is not None
        parent = cast(DictionaryObject, ind.get_object()).get('/Parent', None)
        while parent is not None:
            parent = cast(DictionaryObject, parent.get_object())
            try:
                i = parent['/Kids'].index(ind)
                del parent['/Kids'][i]
                try:
                    assert ind is not None
                    del ind.pdf.flattened_pages[index]
                except Exception:
                    pass
                if '/Count' in parent:
                    parent[NameObject('/Count')] = NumberObject(parent['/Count'] - 1)
                if len(parent['/Kids']) == 0:
                    ind = parent.indirect_reference
                    parent = cast(DictionaryObject, parent.get('/Parent', None))
                else:
                    parent = None
            except ValueError:
                raise PdfReadError(f'Page Not Found in Page Tree {ind}')

    def __iter__(self) -> Iterator[PageObject]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        p = [f'PageObject({i})' for i in range(self.length_function())]
        return f'[{', '.join(p)}]'