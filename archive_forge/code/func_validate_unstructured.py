import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def validate_unstructured(self) -> None:
    assert not self.structured, 'This function is structured, but there was no valid functional variant of it.'
    assert self.structured_delegate, 'This function delegates to another structured out function, but no valid function was found (the delegate may not exist, or it has the wrong type)'