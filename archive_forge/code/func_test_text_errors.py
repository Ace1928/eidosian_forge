import unittest
import os
import contextlib
import importlib_resources as resources
def test_text_errors(self):
    for func in (resources.read_text, resources.open_text):
        with self.subTest(func=func):
            with self.assertRaises(TypeError):
                func(self.anchor02, 'subdirectory', 'subsubdir', 'resource.txt')