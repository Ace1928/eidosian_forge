import collections
import copy
import itertools
import random
import re
import warnings
def total_branch_length(self):
    """Calculate the sum of all the branch lengths in this tree."""
    return sum((node.branch_length for node in self.find_clades(branch_length=True)))