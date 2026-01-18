import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_short_tokens(self):
    self.assertEqual(remove_short_tokens(['salut', 'les', 'amis', 'du', '59'], 3), ['salut', 'les', 'amis'])