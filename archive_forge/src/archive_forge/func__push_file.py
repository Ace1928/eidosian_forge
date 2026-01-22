import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def _push_file(self, dict_, fname, line_no):
    if fname not in dict_:
        dict_[fname] = [line_no]
    else:
        dict_[fname].append(line_no)