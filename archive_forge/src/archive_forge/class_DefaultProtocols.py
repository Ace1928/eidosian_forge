import sys
import pickle
import typing
import contextlib
from aiokeydb.v2.types import BaseSerializer
class DefaultProtocols:
    default: int = 4
    pickle: int = pickle.HIGHEST_PROTOCOL
    dill: int = dill.HIGHEST_PROTOCOL