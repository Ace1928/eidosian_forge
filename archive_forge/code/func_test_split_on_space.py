import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_split_on_space(self):
    self.assertEqual(split_on_space(' salut   les  amis du 59 '), ['salut', 'les', 'amis', 'du', '59'])