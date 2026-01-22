import unittest
import unittest.mock
import os
from pathlib import Path
from bpython.translations import init
class FixLanguageTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init(languages=['en'])