import functools
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import tree
from taskflow.utils import misc
def _cached_get(cache, cache_key, atom_name, fetch_func, *args, **kwargs):
    """Tries to get a previously saved value or fetches it and caches it."""
    value, value_found = (None, False)
    try:
        value, value_found = cache[cache_key][atom_name]
    except KeyError:
        try:
            value = fetch_func(*args, **kwargs)
            value_found = True
        except (exc.StorageFailure, exc.NotFound):
            pass
        cache[cache_key][atom_name] = (value, value_found)
    return (value, value_found)