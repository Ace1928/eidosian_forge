import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def assertFilesDeleted(self, files):
    for f in files:
        id = f.encode('utf-8') + _id
        self.assertNotInWorkingTree(f)
        self.assertPathDoesNotExist(f)