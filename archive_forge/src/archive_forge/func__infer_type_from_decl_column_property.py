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
def _infer_type_from_decl_column_property(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, node: Var, left_hand_explicit_type: Optional[ProperType]) -> Optional[ProperType]:
    """Infer the type of mapping from a ColumnProperty.

    This includes mappings against ``column_property()`` as well as the
    ``deferred()`` function.

    """
    assert isinstance(stmt.rvalue, CallExpr)
    if stmt.rvalue.args:
        first_prop_arg = stmt.rvalue.args[0]
        if isinstance(first_prop_arg, CallExpr):
            type_id = names.type_id_for_callee(first_prop_arg.callee)
            if type_id is names.COLUMN:
                return _infer_type_from_decl_column(api, stmt, node, left_hand_explicit_type, right_hand_expression=first_prop_arg)
    if isinstance(stmt.rvalue, CallExpr):
        type_id = names.type_id_for_callee(stmt.rvalue.callee)
        if type_id is names.QUERY_EXPRESSION:
            return _infer_type_from_decl_column(api, stmt, node, left_hand_explicit_type)
    return infer_type_from_left_hand_type_only(api, node, left_hand_explicit_type)