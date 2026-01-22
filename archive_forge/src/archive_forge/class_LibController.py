import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
class LibController(ABC):
    """Abstract base class for the individual library controllers

    A library controller must expose the following class attributes:
        - user_api : str
            Usually the name of the library or generic specification the library
            implements, e.g. "blas" is a specification with different implementations.
        - internal_api : str
            Usually the name of the library or concrete implementation of some
            specification, e.g. "openblas" is an implementation of the "blas"
            specification.
        - filename_prefixes : tuple
            Possible prefixes of the shared library's filename that allow to
            identify the library. e.g. "libopenblas" for libopenblas.so.

    and implement the following methods: `get_num_threads`, `set_num_threads` and
    `get_version`.

    Threadpoolctl loops through all the loaded shared libraries and tries to match
    the filename of each library with the `filename_prefixes`. If a match is found, a
    controller is instantiated and a handler to the library is stored in the `dynlib`
    attribute as a `ctypes.CDLL` object. It can be used to access the necessary symbols
    of the shared library to implement the above methods.

    The following information will be exposed in the info dictionary:
      - user_api : standardized API, if any, or a copy of internal_api.
      - internal_api : implementation-specific API.
      - num_threads : the current thread limit.
      - prefix : prefix of the shared library's filename.
      - filepath : path to the loaded shared library.
      - version : version of the library (if available).

    In addition, each library controller may expose internal API specific entries. They
    must be set as attributes in the `set_additional_attributes` method.
    """

    @final
    def __init__(self, *, filepath=None, prefix=None, parent=None):
        """This is not meant to be overriden by subclasses."""
        self.parent = parent
        self.prefix = prefix
        self.filepath = filepath
        self.dynlib = ctypes.CDLL(filepath, mode=_RTLD_NOLOAD)
        self.version = self.get_version()
        self.set_additional_attributes()

    def info(self):
        """Return relevant info wrapped in a dict"""
        exposed_attrs = {'user_api': self.user_api, 'internal_api': self.internal_api, 'num_threads': self.num_threads, **vars(self)}
        exposed_attrs.pop('dynlib')
        exposed_attrs.pop('parent')
        return exposed_attrs

    def set_additional_attributes(self):
        """Set additional attributes meant to be exposed in the info dict"""

    @property
    def num_threads(self):
        """Exposes the current thread limit as a dynamic property

        This is not meant to be used or overriden by subclasses.
        """
        return self.get_num_threads()

    @abstractmethod
    def get_num_threads(self):
        """Return the maximum number of threads available to use"""

    @abstractmethod
    def set_num_threads(self, num_threads):
        """Set the maximum number of threads to use"""

    @abstractmethod
    def get_version(self):
        """Return the version of the shared library"""