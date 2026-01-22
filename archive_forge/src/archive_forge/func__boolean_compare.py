from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import type_api
from .elements import and_
from .elements import BinaryExpression
from .elements import ClauseElement
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ExpressionClauseList
from .elements import False_
from .elements import Null
from .elements import OperatorExpression
from .elements import or_
from .elements import True_
from .elements import UnaryExpression
from .operators import OperatorType
from .. import exc
from .. import util
def _boolean_compare(expr: ColumnElement[Any], op: OperatorType, obj: Any, *, negate_op: Optional[OperatorType]=None, reverse: bool=False, _python_is_types: Tuple[Type[Any], ...]=(type(None), bool), result_type: Optional[TypeEngine[bool]]=None, **kwargs: Any) -> OperatorExpression[bool]:
    if result_type is None:
        result_type = type_api.BOOLEANTYPE
    if isinstance(obj, _python_is_types + (Null, True_, False_)):
        if op in (operators.eq, operators.ne) and isinstance(obj, (bool, True_, False_)):
            return OperatorExpression._construct_for_op(expr, coercions.expect(roles.ConstExprRole, obj), op, type_=result_type, negate=negate_op, modifiers=kwargs)
        elif op in (operators.is_distinct_from, operators.is_not_distinct_from):
            return OperatorExpression._construct_for_op(expr, coercions.expect(roles.ConstExprRole, obj), op, type_=result_type, negate=negate_op, modifiers=kwargs)
        elif expr._is_collection_aggregate:
            obj = coercions.expect(roles.ConstExprRole, element=obj, operator=op, expr=expr)
        elif op in (operators.eq, operators.is_):
            return OperatorExpression._construct_for_op(expr, coercions.expect(roles.ConstExprRole, obj), operators.is_, negate=operators.is_not, type_=result_type)
        elif op in (operators.ne, operators.is_not):
            return OperatorExpression._construct_for_op(expr, coercions.expect(roles.ConstExprRole, obj), operators.is_not, negate=operators.is_, type_=result_type)
        else:
            raise exc.ArgumentError("Only '=', '!=', 'is_()', 'is_not()', 'is_distinct_from()', 'is_not_distinct_from()' operators can be used with None/True/False")
    else:
        obj = coercions.expect(roles.BinaryElementRole, element=obj, operator=op, expr=expr)
    if reverse:
        return OperatorExpression._construct_for_op(obj, expr, op, type_=result_type, negate=negate_op, modifiers=kwargs)
    else:
        return OperatorExpression._construct_for_op(expr, obj, op, type_=result_type, negate=negate_op, modifiers=kwargs)