import os
import shutil
import sys
import tempfile
import unittest
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
def test_w_mode(self):
    with open(self.tmpfilepath, 'w') as f:
        f.write(self.teststring)
        f.seek(0)
        self.assertRaises(pickle.PicklingError, lambda: cloudpickle.dumps(f))
    os.remove(self.tmpfilepath)