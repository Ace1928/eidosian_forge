from asyncio import sleep
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Tuple, Type, TypeVar
from aiokeydb.v1.exceptions import ConnectionError, KeyDBError, TimeoutError
def update_supported_errors(self, specified_errors: list):
    """
        Updates the supported errors with the specified error types
        """
    self._supported_errors = tuple(set(self._supported_errors + tuple(specified_errors)))