import unittest
import contextlib
import pathlib
import importlib_resources as resources
from .. import abc
from ..abc import TraversableResources, ResourceReader
from . import util
from .compat.py39 import os_helper
class MagicResources(TraversableResources):
    """
    Magically returns the resources at path.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path

    def files(self):
        return self.path