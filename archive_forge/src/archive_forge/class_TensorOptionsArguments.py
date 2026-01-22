import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class TensorOptionsArguments:
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]