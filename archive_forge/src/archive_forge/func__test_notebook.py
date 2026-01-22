from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def _test_notebook(self, notebook, test):
    with open(notebook) as f:
        nb = convert(reads(f.read()), self.NBFORMAT_VERSION)
    _, kernel = utils.start_new_kernel()
    for i, c in enumerate([c for c in nb.cells if c.cell_type == 'code']):
        self._test_notebook_cell(self.sanitize_cell(c), i, kernel, test)