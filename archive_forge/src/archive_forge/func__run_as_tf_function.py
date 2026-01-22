import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def _run_as_tf_function(self, fn):

    def wrapper(self):

        @def_function.function(autograph=False)
        def fn_wrapper():
            self.assertions = []
            self.raises_cm = None
            self.graph_assertions = []
            self.trace_log = []
            fn()
            targets = [args for _, args in self.assertions]
            return targets
        try:
            tensors = fn_wrapper()
            for assertion in self.graph_assertions:
                assertion(fn_wrapper.get_concrete_function().graph)
            actuals = self.evaluate(tensors)
        except:
            if self.raises_cm is not None:
                self.raises_cm.__exit__(*sys.exc_info())
                return
            else:
                raise
        for (assertion, _), values in zip(self.assertions, actuals):
            assertion(*values)
    return wrapper