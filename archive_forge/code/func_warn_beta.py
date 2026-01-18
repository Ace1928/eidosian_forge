import functools
import warnings
from typing import Callable
def warn_beta(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f'Function {func.__name__} is in beta.', UserWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper