from collections.abc import Sequence
from typing import (
import numpy as np
from ._shape import _ShapeLike
from ._char_codes import (
class _DTypeDict(_DTypeDictBase, total=False):
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool