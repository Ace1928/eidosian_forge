import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_split_alphanum(self):
    self.assertEqual(split_alphanum('toto diet1 titi'), 'toto diet 1 titi')
    self.assertEqual(split_alphanum('toto 1diet titi'), 'toto 1 diet titi')