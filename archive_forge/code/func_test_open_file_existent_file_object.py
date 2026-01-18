import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_open_file_existent_file_object(self):
    number_of_lines_in_file = 30
    file_obj = open(datapath('testcorpus.mm'))
    with utils.open_file(file_obj) as infile:
        self.assertEqual(sum((1 for _ in infile)), number_of_lines_in_file)