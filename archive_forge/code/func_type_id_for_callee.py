from __future__ import annotations
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from mypy.nodes import ARG_POS
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import Decorator
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import OverloadedFuncDef
from mypy.nodes import SymbolNode
from mypy.nodes import TypeAlias
from mypy.nodes import TypeInfo
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import UnboundType
from ... import util
def type_id_for_callee(callee: Expression) -> Optional[int]:
    if isinstance(callee, (MemberExpr, NameExpr)):
        if isinstance(callee.node, Decorator) and isinstance(callee.node.func, FuncDef):
            if callee.node.func.type and isinstance(callee.node.func.type, CallableType):
                ret_type = get_proper_type(callee.node.func.type.ret_type)
                if isinstance(ret_type, Instance):
                    return type_id_for_fullname(ret_type.type.fullname)
            return None
        elif isinstance(callee.node, OverloadedFuncDef):
            if callee.node.impl and callee.node.impl.type and isinstance(callee.node.impl.type, CallableType):
                ret_type = get_proper_type(callee.node.impl.type.ret_type)
                if isinstance(ret_type, Instance):
                    return type_id_for_fullname(ret_type.type.fullname)
            return None
        elif isinstance(callee.node, FuncDef):
            if callee.node.type and isinstance(callee.node.type, CallableType):
                ret_type = get_proper_type(callee.node.type.ret_type)
                if isinstance(ret_type, Instance):
                    return type_id_for_fullname(ret_type.type.fullname)
            return None
        elif isinstance(callee.node, TypeAlias):
            target_type = get_proper_type(callee.node.target)
            if isinstance(target_type, Instance):
                return type_id_for_fullname(target_type.type.fullname)
        elif isinstance(callee.node, TypeInfo):
            return type_id_for_named_node(callee)
    return None