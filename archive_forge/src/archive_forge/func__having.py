from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def _having(as_where: bool=False) -> str:
    if having is None:
        return ''
    pre = 'WHERE ' if as_where else 'HAVING '
    return pre + self.generate(having.alias(''))