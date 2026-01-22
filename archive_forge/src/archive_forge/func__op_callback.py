import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def _op_callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
    self.trace_log.append(op_type)