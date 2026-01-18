from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def type_to_expr(self, data_type: pa.DataType):
    return _type_to_expression(data_type)