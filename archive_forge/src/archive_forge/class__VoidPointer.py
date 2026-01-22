import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
class _VoidPointer(object):

    @abc.abstractmethod
    def get(self):
        """Return the memory location we point to"""
        return

    @abc.abstractmethod
    def address_of(self):
        """Return a raw pointer to this pointer"""
        return