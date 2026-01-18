import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
Test that 'brz revision-info' honors the '-d' option.