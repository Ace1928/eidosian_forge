import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar


        Args:
            latch: If True, we set a flag the first time the group is flushed;
                we then immediately flush any futures added after that point.
                If False, the default, we store all added futures in a list and
                flush them the next time the group is flushed, regardless of
                whether the group has been flushed before.
        