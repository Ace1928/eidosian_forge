import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
class UnsupportedDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(UnsupportedDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = None
        self.backend = None