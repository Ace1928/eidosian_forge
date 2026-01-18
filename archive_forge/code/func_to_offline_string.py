from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def to_offline_string(self, statement_cache: MutableMapping[Any, str], statement: ClauseElement, parameters: _CoreSingleExecuteParams) -> str:
    """Generate an "offline string" form of this :class:`.CacheKey`

        The "offline string" is basically the string SQL for the
        statement plus a repr of the bound parameter values in series.
        Whereas the :class:`.CacheKey` object is dependent on in-memory
        identities in order to work as a cache key, the "offline" version
        is suitable for a cache that will work for other processes as well.

        The given ``statement_cache`` is a dictionary-like object where the
        string form of the statement itself will be cached.  This dictionary
        should be in a longer lived scope in order to reduce the time spent
        stringifying statements.


        """
    if self.key not in statement_cache:
        statement_cache[self.key] = sql_str = str(statement)
    else:
        sql_str = statement_cache[self.key]
    if not self.bindparams:
        param_tuple = tuple((parameters[key] for key in sorted(parameters)))
    else:
        param_tuple = tuple((parameters.get(bindparam.key, bindparam.value) for bindparam in self.bindparams))
    return repr((sql_str, param_tuple))