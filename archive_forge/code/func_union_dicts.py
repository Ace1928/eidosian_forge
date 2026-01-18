import functools
import inspect
from textwrap import dedent
def union_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result