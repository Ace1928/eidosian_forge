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
def argsnkwargs_to_str(args, kwargs):
    buf = [str(a) for a in tuple(args)]
    buf.extend(['{}={}'.format(k, v) for k, v in kwargs.items()])
    return ', '.join(buf)