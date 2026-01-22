import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
class NativeRPCServer(RPCServer):
    """Native RPC Server that can only be used locally.

    :param conf: |FugueConfig|
    """

    def make_client(self, handler: Any) -> RPCClient:
        """Add ``handler`` and correspondent :class:`~.NativeRPCClient`

        :param handler: |RPCHandlerLikeObject|
        :return: the native RPC client
        """
        key = self.register(handler)
        return NativeRPCClient(self, key)

    def start_server(self) -> None:
        """Do nothing"""
        return

    def stop_server(self) -> None:
        """Do nothing"""
        return