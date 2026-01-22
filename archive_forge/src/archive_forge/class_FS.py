import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
class FS:
    exists = staticmethod(os.path.exists)

    @staticmethod
    def open(name, mode='r', **kwargs):
        if _has_atomicwrites and 'w' in mode:
            return atomicwrites.atomic_write(name, mode=mode, overwrite=True, **kwargs)
        else:
            return open(name, mode, **kwargs)