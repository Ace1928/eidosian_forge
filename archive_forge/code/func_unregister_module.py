import sys
import pickle
import typing
import binascii
import contextlib
from io import BytesIO
from aiokeydb.types.serializer import BaseSerializer
from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
@staticmethod
def unregister_module(module: ModuleType):
    """
            Registers a class with cloudpickle
            """
    cloudpickle.unregister_pickle_by_value(module)