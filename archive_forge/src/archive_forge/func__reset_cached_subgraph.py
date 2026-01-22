import collections
import functools
from taskflow import deciders as de
from taskflow import exceptions as exc
from taskflow import flow
from taskflow.types import graph as gr
def _reset_cached_subgraph(func):
    """Resets cached subgraph after execution, in case it was affected."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._subgraph = None
        return result
    return wrapper