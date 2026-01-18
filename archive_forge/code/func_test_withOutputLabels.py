from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_withOutputLabels(self):
    """
        L{tableMaker} adds a colspan attribute to the input label's cell
        equal to the number of output labels and a second row that
        contains the output labels.
        """
    table = self.tableMaker(self.inputLabel, ('output label 1', 'output label 2'), port=self.port)
    self.assertEqual(len(table.children), 2)
    inputRow, outputRow = table.children

    def hasCorrectColspan(element):
        return not isLeaf(element) and element.name == 'td' and (element.attributes.get('colspan') == '2')
    self.assertEqual(len(findElements(inputRow, hasCorrectColspan)), 1)
    self.assertEqual(findElements(outputRow, isLeaf), ['output label 1', 'output label 2'])