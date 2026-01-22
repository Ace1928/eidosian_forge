import inspect
from typing import Dict, Any
def _make_default_values(fullargspec: inspect.FullArgSpec) -> Dict[str, Any]:
    """Returns default values from the function's fullargspec."""
    if fullargspec.defaults is not None:
        defaults = {name: value for name, value in zip(fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)}
    else:
        defaults = {}
    if fullargspec.kwonlydefaults is not None:
        defaults.update(fullargspec.kwonlydefaults)
    return defaults