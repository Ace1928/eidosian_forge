import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_submodule_contents(self):
    contents = names(resources.files(import_module('namespacedata01')))
    try:
        contents.remove('__pycache__')
    except KeyError:
        pass
    self.assertEqual(contents, {'subdirectory', 'binary.file', 'utf-8.file', 'utf-16.file'})