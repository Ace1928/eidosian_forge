import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def assertRaisesRuntime(self, *args):
    if self.raises_cm is not None:
        raise ValueError('cannot use more than one assertRaisesRuntime in a test')
    self.raises_cm = self.assertRaisesRegex(*args)
    self.raises_cm.__enter__()