import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def unambiguous_name(self) -> str:
    if self.overload_name:
        return f'{self.name}_{self.overload_name}'
    else:
        return f'{self.name}'