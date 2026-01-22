import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def _iterate_cases(generate_test):

    def wrap(fn):

        def wrapped(self):
            return generate_test(self, fn)
        wrapped.__doc__ = f'generated test for {fn.__module__}.{fn.__name__}'
        return wrapped
    for k, case_fn in proto.__dict__.items():
        if k.startswith('case'):
            yield (f'test_{k}', wrap(case_fn))