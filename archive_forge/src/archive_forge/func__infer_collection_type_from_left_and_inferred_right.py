from __future__ import annotations
from typing import Optional
from typing import Sequence
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import LambdaExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.subtypes import is_subtype
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnionType
from . import names
from . import util
def _infer_collection_type_from_left_and_inferred_right(api: SemanticAnalyzerPluginInterface, node: Var, left_hand_explicit_type: Instance, python_type_for_type: Instance) -> Optional[ProperType]:
    orig_left_hand_type = left_hand_explicit_type
    orig_python_type_for_type = python_type_for_type
    if left_hand_explicit_type.args:
        left_hand_arg = get_proper_type(left_hand_explicit_type.args[0])
        python_type_arg = get_proper_type(python_type_for_type.args[0])
    else:
        left_hand_arg = left_hand_explicit_type
        python_type_arg = python_type_for_type
    assert isinstance(left_hand_arg, (Instance, UnionType))
    assert isinstance(python_type_arg, (Instance, UnionType))
    return _infer_type_from_left_and_inferred_right(api, node, left_hand_arg, python_type_arg, orig_left_hand_type=orig_left_hand_type, orig_python_type_for_type=orig_python_type_for_type)