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
def _between_impl(expr: ColumnElement[Any], op: OperatorType, cleft: Any, cright: Any, **kw: Any) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.between`."""
    return BinaryExpression(expr, ExpressionClauseList._construct_for_list(operators.and_, type_api.NULLTYPE, coercions.expect(roles.BinaryElementRole, cleft, expr=expr, operator=operators.and_), coercions.expect(roles.BinaryElementRole, cright, expr=expr, operator=operators.and_), group=False), op, negate=operators.not_between_op if op is operators.between_op else operators.between_op, modifiers=kw)