import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
def _unlit_non_poison(ty):
    """Apply unliteral(ty) and raise a TypingError if type is Poison.
    """
    out = unliteral(ty)
    if isinstance(out, types.Poison):
        m = f'Poison type used in arguments; got {out}'
        raise errors.TypingError(m)
    return out