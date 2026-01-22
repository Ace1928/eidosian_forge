from __future__ import annotations
import functools
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import coercions
from . import dml
from . import elements
from . import roles
from .base import _DefaultDescriptionTuple
from .dml import isinsert as _compile_state_isinsert
from .elements import ColumnClause
from .schema import default_is_clause_element
from .schema import default_is_sequence
from .selectable import Select
from .selectable import TableClause
from .. import exc
from .. import util
from ..util.typing import Literal
def _create_update_prefetch_bind_param(compiler: SQLCompiler, c: ColumnElement[Any], process: bool=True, name: Optional[str]=None, **kw: Any) -> Union[elements.BindParameter[Any], str]:
    param = _create_bind_param(compiler, c, None, process=process, name=name, **kw)
    compiler.update_prefetch.append(c)
    return param