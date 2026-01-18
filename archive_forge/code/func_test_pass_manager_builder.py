import unittest
from ..pass_manager import (
def test_pass_manager_builder(self) -> None:
    passes = [lambda x: 2 * x for _ in range(10)]
    pm = PassManager(passes)
    pm.validate()