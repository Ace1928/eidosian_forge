import unittest
from ..pass_manager import (
def test_these_before_those_pass_constraint(self) -> None:
    passes = [lambda x: 2 * x for _ in range(10)]
    constraint = these_before_those_pass_constraint(passes[-1], passes[0])
    pm = PassManager([inplace_wrapper(p) for p in passes])
    pm.add_constraint(constraint)
    self.assertRaises(RuntimeError, pm.validate)