from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_typednames_correct_forward_ref(self):
    self.flakes('\n            from typing import TypedDict, List, NamedTuple\n\n            List[TypedDict("x", {})]\n            List[TypedDict("x", x=int)]\n            List[NamedTuple("a", a=int)]\n            List[NamedTuple("a", [("a", int)])]\n        ')
    self.flakes('\n            from typing import TypedDict, List, NamedTuple, TypeVar\n\n            List[TypedDict("x", {"x": "Y"})]\n            List[TypedDict("x", x="Y")]\n            List[NamedTuple("a", [("a", "Y")])]\n            List[NamedTuple("a", a="Y")]\n            List[TypedDict("x", {"x": List["a"]})]\n            List[TypeVar("A", bound="C")]\n            List[TypeVar("A", List["C"])]\n        ', *[m.UndefinedName] * 7)
    self.flakes('\n            from typing import NamedTuple, TypeVar, cast\n            from t import A, B, C, D, E\n\n            NamedTuple("A", [("a", A["C"])])\n            TypeVar("A", bound=A["B"])\n            TypeVar("A", A["D"])\n            cast(A["E"], [])\n        ')