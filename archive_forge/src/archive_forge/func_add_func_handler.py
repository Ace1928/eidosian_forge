from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def add_func_handler(self, name: str, handler: Callable[[_FuncExpr], Iterable[str]]) -> 'SQLExpressionGenerator':
    """Add special function handler.

        :param name: name of the function
        :param handler: the function to convert the function expression to SQL
          clause
        :return: the instance itself

        .. caution::

            Users should not use this directly
        """
    self._func_handler[name] = handler
    return self