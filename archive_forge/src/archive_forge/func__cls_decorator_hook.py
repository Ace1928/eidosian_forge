from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type as TypingType
from typing import Union
from mypy import nodes
from mypy.mro import calculate_mro
from mypy.mro import MroError
from mypy.nodes import Block
from mypy.nodes import ClassDef
from mypy.nodes import GDEF
from mypy.nodes import MypyFile
from mypy.nodes import NameExpr
from mypy.nodes import SymbolTable
from mypy.nodes import SymbolTableNode
from mypy.nodes import TypeInfo
from mypy.plugin import AttributeContext
from mypy.plugin import ClassDefContext
from mypy.plugin import DynamicClassDefContext
from mypy.plugin import Plugin
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import Type
from . import decl_class
from . import names
from . import util
def _cls_decorator_hook(ctx: ClassDefContext) -> None:
    _add_globals(ctx)
    assert isinstance(ctx.reason, nodes.MemberExpr)
    expr = ctx.reason.expr
    assert isinstance(expr, nodes.RefExpr) and isinstance(expr.node, nodes.Var)
    node_type = get_proper_type(expr.node.type)
    assert isinstance(node_type, Instance) and names.type_id_for_named_node(node_type.type) is names.REGISTRY
    decl_class.scan_declarative_assignments_and_apply_types(ctx.cls, ctx.api)