import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
def try_set_result(self, result: T) -> bool:
    """Sets the result on this future if not already done.

        Returns:
            True if we set the result, False if the future was already done.
        """
    with self._condition:
        if self.done():
            return False
        self.set_result(result)
        return True