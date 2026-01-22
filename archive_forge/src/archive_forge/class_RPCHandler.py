import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
class RPCHandler(RPCClient):
    """RPC handler hosting the real logic on driver side"""

    def __init__(self):
        self._rpchandler_lock = SerializableRLock()
        self._running = 0

    @property
    def running(self) -> bool:
        """Whether the handler is in running state"""
        return self._running > 0

    def __uuid__(self) -> str:
        """UUID that can affect the determinism of the workflow"""
        return ''

    def start_handler(self) -> None:
        """User implementation of starting the handler"""
        return

    def stop_handler(self) -> None:
        """User implementation of stopping the handler"""
        return

    def start(self) -> 'RPCHandler':
        """Start the handler, wrapping :meth:`~.start_handler`

        :return: the instance itself
        """
        with self._rpchandler_lock:
            if self._running == 0:
                self.start_handler()
            self._running += 1
        return self

    def stop(self) -> None:
        """Stop the handler, wrapping :meth:`~.stop_handler`"""
        with self._rpchandler_lock:
            if self._running == 1:
                self.stop_handler()
            self._running -= 1
            if self._running < 0:
                self._running = 0

    def __enter__(self) -> 'RPCHandler':
        """``with`` statement. :meth:`~.start` must be called

        .. admonition:: Examples

            .. code-block:: python

                with handler.start():
                    handler...
        """
        with self._rpchandler_lock:
            assert_or_throw(self._running, 'use `with <instance>.start():` instead')
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __getstate__(self):
        """
        :raises pickle.PicklingError: serialization
          of ``RPCHandler`` is not allowed
        """
        raise pickle.PicklingError(f'{self} is not serializable')

    def __copy__(self) -> 'RPCHandler':
        """Copy takes no effect

        :return: the instance itself
        """
        return self

    def __deepcopy__(self, memo: Any) -> 'RPCHandler':
        """Deep copy takes no effect

        :return: the instance itself
        """
        return self