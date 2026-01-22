from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, cast
import numpy
from thinc.api import (
from thinc.loss import Loss
from thinc.types import Floats2d, Ints1d
from ...attrs import ID, ORTH
from ...errors import Errors
from ...util import OOV_RANK, registry
from ...vectors import Mode as VectorsMode
def _replace_word(word: str, random_words: _RandomWords, mask: str='[MASK]') -> str:
    roll = numpy.random.random()
    if roll < 0.8:
        return mask
    elif roll < 0.9:
        return random_words.next()
    else:
        return word