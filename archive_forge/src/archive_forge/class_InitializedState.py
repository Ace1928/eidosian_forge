import io
import posixpath
import zipfile
import itertools
import contextlib
import pathlib
import re
import sys
from .compat.py310 import text_encoding
from .glob import Translator
class InitializedState:
    """
    Mix-in to save the initialization state for pickling.
    """

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return (self.__args, self.__kwargs)

    def __setstate__(self, state):
        args, kwargs = state
        super().__init__(*args, **kwargs)