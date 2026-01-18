import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_short(self):
    self.assertEqual(strip_short('salut les amis du 59', 3), 'salut les amis')