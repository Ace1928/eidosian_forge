import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def variable(self, name, value, dtype):
    with ops.init_scope():
        if name not in self.variables:
            self.variables[name] = variables.Variable(value, dtype=dtype)
            self.evaluate(self.variables[name].initializer)
    return self.variables[name]