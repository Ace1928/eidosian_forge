import functools
from typing import Callable, Generator, Iterable, Iterator, Optional, Tuple
from pip._vendor.rich.progress import (
from pip._internal.utils.logging import get_indentation
Get an object that can be used to render the download progress.

    Returns a callable, that takes an iterable to "wrap".
    