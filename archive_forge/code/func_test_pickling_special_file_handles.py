import os
import shutil
import sys
import tempfile
import unittest
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
@pytest.mark.skip(reason='Requires pytest -s to pass')
def test_pickling_special_file_handles(self):
    for out in (sys.stdout, sys.stderr):
        self.assertEqual(out, pickle.loads(cloudpickle.dumps(out)))
    self.assertRaises(pickle.PicklingError, lambda: cloudpickle.dumps(sys.stdin))