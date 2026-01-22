import sys
import pickle
import typing
import contextlib
from aiokeydb.v2.types import BaseSerializer
class DillSerializer(BaseSerializer):

    @staticmethod
    def dumps(obj: typing.Any, protocol: int=DefaultProtocols.dill, *args, **kwargs) -> bytes:
        return dill.dumps(obj, *args, protocol=protocol, **kwargs)

    @staticmethod
    def loads(data: typing.Union[str, bytes, typing.Any], *args, **kwargs) -> typing.Any:
        return dill.loads(data, *args, **kwargs)