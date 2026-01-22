import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
class TestApiUsage(TestSourceHelper):

    def find_occurences(self, rule, filename):
        """Find the number of occurences of rule in a file."""
        occurences = 0
        source = open(filename)
        for line in source:
            if line.find(rule) > -1:
                occurences += 1
        return occurences

    def test_branch_working_tree(self):
        """Test that the number of uses of working_tree in branch is stable."""
        occurences = self.find_occurences('self.working_tree()', self.source_file_name(breezy.branch))
        self.assertEqual(0, occurences)

    def test_branch_WorkingTree(self):
        """Test that the number of uses of working_tree in branch is stable."""
        occurences = self.find_occurences('WorkingTree', self.source_file_name(breezy.branch))
        self.assertEqual(0, occurences)