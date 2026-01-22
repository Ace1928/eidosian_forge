import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class OptionalType(Type):
    elem: Type

    def __str__(self) -> str:
        return f'{self.elem}?'

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        return self.elem.is_base_ty_like(base_ty)

    def is_symint_like(self) -> bool:
        return self.elem.is_symint_like()

    def is_nullable(self) -> bool:
        return True

    def is_list_like(self) -> Optional['ListType']:
        return self.elem.is_list_like()