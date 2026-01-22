import os
from typing import Callable, Generator, Union
def _include_fn(path: str, root: str) -> bool:
    return include_fn(path, root) if len(inspect.signature(include_fn).parameters) == 2 else include_fn(path)