from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
class Crypter:
    """Base class providing encryption and decryption"""

    @abc.abstractmethod
    def encrypt(self, value):
        """Encrypt the value."""
        pass

    @abc.abstractmethod
    def decrypt(self, value):
        """Decrypt the value."""
        pass