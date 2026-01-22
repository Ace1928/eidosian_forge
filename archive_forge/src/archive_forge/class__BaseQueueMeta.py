import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class _BaseQueueMeta(type):
    """
    Metaclass to check queue classes against the necessary interface
    """

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return hasattr(subclass, 'push') and callable(subclass.push) and hasattr(subclass, 'pop') and callable(subclass.pop) and hasattr(subclass, 'peek') and callable(subclass.peek) and hasattr(subclass, 'close') and callable(subclass.close) and hasattr(subclass, '__len__') and callable(subclass.__len__)