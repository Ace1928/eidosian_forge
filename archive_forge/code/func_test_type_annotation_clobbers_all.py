from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_type_annotation_clobbers_all(self):
    self.flakes('        from typing import TYPE_CHECKING, List\n\n        from y import z\n\n        if not TYPE_CHECKING:\n            __all__ = ("z",)\n        else:\n            __all__: List[str]\n        ')