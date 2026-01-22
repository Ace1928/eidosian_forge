import warnings
from functools import wraps
from typing import Callable
from .. import constants
Decorator to flag a feature as experimental.

    An experimental feature trigger a warning when used as it might be subject to breaking changes in the future.
    Warnings can be disabled by setting the environment variable `HF_EXPERIMENTAL_WARNING` to `0`.

    Args:
        fn (`Callable`):
            The function to flag as experimental.

    Returns:
        `Callable`: The decorated function.

    Example:

    ```python
    >>> from huggingface_hub.utils import experimental

    >>> @experimental
    ... def my_function():
    ...     print("Hello world!")

    >>> my_function()
    UserWarning: 'my_function' is experimental and might be subject to breaking changes in the future. You can disable
    this warning by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as environment variable.
    Hello world!
    ```
    