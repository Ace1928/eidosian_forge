import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_stem_text(self):
    target = 'while it is quit us to be abl to search a larg ' + 'collect of document almost instantli for a joint occurr ' + 'of a collect of exact words, for mani search purposes, ' + 'a littl fuzzi would help.'
    self.assertEqual(stem_text(doc5), target)