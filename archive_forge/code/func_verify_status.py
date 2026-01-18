import os
from io import StringIO
from .. import errors
from ..status import show_tree_status
from . import TestCaseWithTransport
from .features import OsFifoFeature
def verify_status(tester, tree, value):
    """Verify the output of show_tree_status"""
    tof = StringIO()
    show_tree_status(tree, to_file=tof)
    tof.seek(0)
    tester.assertEqual(value, tof.readlines())