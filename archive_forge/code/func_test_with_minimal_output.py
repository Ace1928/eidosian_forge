import unittest
from charset_normalizer.cli.normalizer import cli_detect, query_yes_no
from unittest.mock import patch
from os.path import exists
from os import remove
def test_with_minimal_output(self):
    self.assertEqual(0, cli_detect(['-m', './data/sample-arabic-1.txt', './data/sample-french.txt', './data/sample-chinese.txt']))