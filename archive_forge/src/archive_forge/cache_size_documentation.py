import logging
import types
import weakref
from dataclasses import dataclass
from . import config

    Checks if we are exceeding the cache size limit.
    