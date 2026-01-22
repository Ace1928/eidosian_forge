import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def _loop_nest(builder, shape, intp):
    with for_range(builder, shape[0], intp=intp) as loop:
        if len(shape) > 1:
            with _loop_nest(builder, shape[1:], intp) as indices:
                yield ((loop.index,) + indices)
        else:
            yield (loop.index,)