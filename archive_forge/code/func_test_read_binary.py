import unittest
import os
import contextlib
import importlib_resources as resources
def test_read_binary(self):
    self.assertEqual(resources.read_binary(self.anchor01, 'utf-8.file'), b'Hello, UTF-8 world!\n')
    for path_parts in self._gen_resourcetxt_path_parts():
        self.assertEqual(resources.read_binary(self.anchor02, *path_parts), b'a resource')