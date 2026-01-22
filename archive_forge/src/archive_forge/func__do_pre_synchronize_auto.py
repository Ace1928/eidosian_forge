from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _do_pre_synchronize_auto(cls, session, statement, params, execution_options, bind_arguments, update_options):
    """setup auto sync strategy


        "auto" checks if we can use "evaluate" first, then falls back
        to "fetch"

        evaluate is vastly more efficient for the common case
        where session is empty, only has a few objects, and the UPDATE
        statement can potentially match thousands/millions of rows.

        OTOH more complex criteria that fails to work with "evaluate"
        we would hope usually correlates with fewer net rows.

        """
    try:
        eval_condition = cls._eval_condition_from_statement(update_options, statement)
    except evaluator.UnevaluatableError:
        pass
    else:
        return update_options + {'_eval_condition': eval_condition, '_synchronize_session': 'evaluate'}
    update_options += {'_synchronize_session': 'fetch'}
    return cls._do_pre_synchronize_fetch(session, statement, params, execution_options, bind_arguments, update_options)