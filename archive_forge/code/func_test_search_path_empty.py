import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_search_path_empty(self):
    loader = TemplateLoader()
    self.assertEqual([], loader.search_path)