import sys
import pickle
import typing
import binascii
import contextlib
from io import BytesIO
from aiokeydb.types.serializer import BaseSerializer
from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
class PickleSerializerv2(BaseSerializer):

    @staticmethod
    def dumps(obj: typing.Any, protocol: int=DEFAULT_PROTOCOL, *args, **kwargs) -> str:
        """
        v2 Encoding
        """
        f = BytesIO()
        p = Pickler(f, protocol=protocol)
        p.dump(obj)
        return f.getvalue().hex()

    @staticmethod
    def loads(data: typing.Union[str, typing.Any], *args, **kwargs) -> typing.Any:
        """
        V2 Decoding
        """
        return Unpickler(BytesIO(binascii.unhexlify(data))).load()