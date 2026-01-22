import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
class NativeRPCClient(RPCClient):
    """Native RPC Client that can only be used locally.
    Use :meth:`~.NativeRPCServer.make_client` to create this instance.

    :param server: the parent :class:`~.NativeRPCServer`
    :param key: the unique key for the handler and this client
    """

    def __init__(self, server: 'NativeRPCServer', key: str):
        self._key = key
        self._server = server

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self._server.invoke(self._key, *args, **kwargs)

    def __getstate__(self):
        raise pickle.PicklingError(f'{self} is not serializable')