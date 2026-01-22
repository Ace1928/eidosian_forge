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
def _infer_type_from_relationship(api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, node: Var, left_hand_explicit_type: Optional[ProperType]) -> Optional[ProperType]:
    """Infer the type of mapping from a relationship.

    E.g.::

        @reg.mapped
        class MyClass:
            # ...

            addresses = relationship(Address, uselist=True)

            order: Mapped["Order"] = relationship("Order")

    Will resolve in mypy as::

        @reg.mapped
        class MyClass:
            # ...

            addresses: Mapped[List[Address]]

            order: Mapped["Order"]

    """
    assert isinstance(stmt.rvalue, CallExpr)
    target_cls_arg = stmt.rvalue.args[0]
    python_type_for_type: Optional[ProperType] = None
    if isinstance(target_cls_arg, NameExpr) and isinstance(target_cls_arg.node, TypeInfo):
        related_object_type = target_cls_arg.node
        python_type_for_type = Instance(related_object_type, [])
    uselist_arg = util.get_callexpr_kwarg(stmt.rvalue, 'uselist')
    collection_cls_arg: Optional[Expression] = util.get_callexpr_kwarg(stmt.rvalue, 'collection_class')
    type_is_a_collection = False
    if uselist_arg is not None and api.parse_bool(uselist_arg) is True and (collection_cls_arg is None):
        type_is_a_collection = True
        if python_type_for_type is not None:
            python_type_for_type = api.named_type(names.NAMED_TYPE_BUILTINS_LIST, [python_type_for_type])
    elif (uselist_arg is None or api.parse_bool(uselist_arg) is True) and collection_cls_arg is not None:
        type_is_a_collection = True
        if isinstance(collection_cls_arg, CallExpr):
            collection_cls_arg = collection_cls_arg.callee
        if isinstance(collection_cls_arg, NameExpr) and isinstance(collection_cls_arg.node, TypeInfo):
            if python_type_for_type is not None:
                python_type_for_type = Instance(collection_cls_arg.node, [python_type_for_type])
        elif isinstance(collection_cls_arg, NameExpr) and isinstance(collection_cls_arg.node, FuncDef) and (collection_cls_arg.node.type is not None):
            if python_type_for_type is not None:
                if isinstance(collection_cls_arg.node.type, CallableType):
                    rt = get_proper_type(collection_cls_arg.node.type.ret_type)
                    if isinstance(rt, CallableType):
                        callable_ret_type = get_proper_type(rt.ret_type)
                        if isinstance(callable_ret_type, Instance):
                            python_type_for_type = Instance(callable_ret_type.type, [python_type_for_type])
        else:
            util.fail(api, 'Expected Python collection type for collection_class parameter', stmt.rvalue)
            python_type_for_type = None
    elif uselist_arg is not None and api.parse_bool(uselist_arg) is False:
        if collection_cls_arg is not None:
            util.fail(api, 'Sending uselist=False and collection_class at the same time does not make sense', stmt.rvalue)
        if python_type_for_type is not None:
            python_type_for_type = UnionType([python_type_for_type, NoneType()])
    elif left_hand_explicit_type is None:
        msg = "Can't infer scalar or collection for ORM mapped expression assigned to attribute '{}' if both 'uselist' and 'collection_class' arguments are absent from the relationship(); please specify a type annotation on the left hand side."
        util.fail(api, msg.format(node.name), node)
    if python_type_for_type is None:
        return infer_type_from_left_hand_type_only(api, node, left_hand_explicit_type)
    elif left_hand_explicit_type is not None:
        if type_is_a_collection:
            assert isinstance(left_hand_explicit_type, Instance)
            assert isinstance(python_type_for_type, Instance)
            return _infer_collection_type_from_left_and_inferred_right(api, node, left_hand_explicit_type, python_type_for_type)
        else:
            return _infer_type_from_left_and_inferred_right(api, node, left_hand_explicit_type, python_type_for_type)
    else:
        return python_type_for_type