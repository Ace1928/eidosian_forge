import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
@contextlib.contextmanager
def _acquire_and_release_futures(fs):
    fs = sorted(fs, key=id)
    with ExitStack() as stack:
        for fut in fs:
            stack.enter_context(fut._condition)
        yield