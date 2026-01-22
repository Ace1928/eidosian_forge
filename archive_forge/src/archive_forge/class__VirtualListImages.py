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
class _VirtualListImages(Sequence[ImageFile]):

    def __init__(self, ids_function: Callable[[], List[Union[str, List[str]]]], get_function: Callable[[Union[str, List[str], Tuple[str]]], ImageFile]) -> None:
        self.ids_function = ids_function
        self.get_function = get_function
        self.current = -1

    def __len__(self) -> int:
        return len(self.ids_function())

    def keys(self) -> List[Union[str, List[str]]]:
        return self.ids_function()

    def items(self) -> List[Tuple[Union[str, List[str]], ImageFile]]:
        return [(x, self[x]) for x in self.ids_function()]

    @overload
    def __getitem__(self, index: Union[int, str, List[str]]) -> ImageFile:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ImageFile]:
        ...

    def __getitem__(self, index: Union[int, slice, str, List[str], Tuple[str]]) -> Union[ImageFile, Sequence[ImageFile]]:
        lst = self.ids_function()
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            lst = [lst[x] for x in indices]
            cls = type(self)
            return cls(lambda: lst, self.get_function)
        if isinstance(index, (str, list, tuple)):
            return self.get_function(index)
        if not isinstance(index, int):
            raise TypeError('invalid sequence indices type')
        len_self = len(lst)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('sequence index out of range')
        return self.get_function(lst[index])

    def __iter__(self) -> Iterator[ImageFile]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        p = [f'Image_{i}={n}' for i, n in enumerate(self.ids_function())]
        return f'[{', '.join(p)}]'