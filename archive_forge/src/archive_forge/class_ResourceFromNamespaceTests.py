import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class ResourceFromNamespaceTests:

    def test_is_submodule_resource(self):
        self.assertTrue(resources.files(import_module('namespacedata01')).joinpath('binary.file').is_file())

    def test_read_submodule_resource_by_name(self):
        self.assertTrue(resources.files('namespacedata01').joinpath('binary.file').is_file())

    def test_submodule_contents(self):
        contents = names(resources.files(import_module('namespacedata01')))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'subdirectory', 'binary.file', 'utf-8.file', 'utf-16.file'})

    def test_submodule_contents_by_name(self):
        contents = names(resources.files('namespacedata01'))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'subdirectory', 'binary.file', 'utf-8.file', 'utf-16.file'})

    def test_submodule_sub_contents(self):
        contents = names(resources.files(import_module('namespacedata01.subdirectory')))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'binary.file'})

    def test_submodule_sub_contents_by_name(self):
        contents = names(resources.files('namespacedata01.subdirectory'))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'binary.file'})