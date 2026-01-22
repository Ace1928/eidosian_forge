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
def _infer_type_from_decl_column(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, node: Var, left_hand_explicit_type: Optional[ProperType], right_hand_expression: Optional[CallExpr]=None) -> Optional[ProperType]:
    """Infer the type of mapping from a Column.

    E.g.::

        @reg.mapped
        class MyClass:
            # ...

            a = Column(Integer)

            b = Column("b", String)

            c: Mapped[int] = Column(Integer)

            d: bool = Column(Boolean)

    Will resolve in MyPy as::

        @reg.mapped
        class MyClass:
            # ...

            a : Mapped[int]

            b : Mapped[str]

            c: Mapped[int]

            d: Mapped[bool]

    """
    assert isinstance(node, Var)
    callee = None
    if right_hand_expression is None:
        if not isinstance(stmt.rvalue, CallExpr):
            return None
        right_hand_expression = stmt.rvalue
    for column_arg in right_hand_expression.args[0:2]:
        if isinstance(column_arg, CallExpr):
            if isinstance(column_arg.callee, RefExpr):
                callee = column_arg.callee
                type_args: Sequence[Expression] = column_arg.args
                break
        elif isinstance(column_arg, (NameExpr, MemberExpr)):
            if isinstance(column_arg.node, TypeInfo):
                callee = column_arg
                type_args = ()
                break
            else:
                continue
        elif isinstance(column_arg, (StrExpr,)):
            continue
        elif isinstance(column_arg, (LambdaExpr,)):
            continue
        else:
            assert False
    if callee is None:
        return None
    if isinstance(callee.node, TypeInfo) and names.mro_has_id(callee.node.mro, names.TYPEENGINE):
        python_type_for_type = extract_python_type_from_typeengine(api, callee.node, type_args)
        if left_hand_explicit_type is not None:
            return _infer_type_from_left_and_inferred_right(api, node, left_hand_explicit_type, python_type_for_type)
        else:
            return UnionType([python_type_for_type, NoneType()])
    else:
        return infer_type_from_left_hand_type_only(api, node, left_hand_explicit_type)