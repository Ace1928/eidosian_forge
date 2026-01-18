import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def view_schema_kind(self) -> ViewSchemaKind:
    if self.is_view_op and self.func.name.name.inplace:
        assert 'inplace_view' in self.tags
        return ViewSchemaKind.aliasing_inplace
    if self.is_view_op:
        return ViewSchemaKind.aliasing
    else:
        return ViewSchemaKind.non_aliasing