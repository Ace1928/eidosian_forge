import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
class NaiveOptimizer(oe.paths.PathOptimizer):

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        self.was_used = True
        return [(0, 1)] * (len(inputs) - 1)