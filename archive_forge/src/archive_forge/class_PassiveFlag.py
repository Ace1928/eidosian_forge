from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class PassiveFlag(FastIntFlag):
    """Bitflag interface that passes options onto loader callables"""
    NO_CHANGE = 0
    'No callables or SQL should be emitted on attribute access\n    and no state should change\n    '
    CALLABLES_OK = 1
    'Loader callables can be fired off if a value\n    is not present.\n    '
    SQL_OK = 2
    'Loader callables can emit SQL at least on scalar value attributes.'
    RELATED_OBJECT_OK = 4
    'Callables can use SQL to load related objects as well\n    as scalar value attributes.\n    '
    INIT_OK = 8
    'Attributes should be initialized with a blank\n    value (None or an empty collection) upon get, if no other\n    value can be obtained.\n    '
    NON_PERSISTENT_OK = 16
    'Callables can be emitted if the parent is not persistent.'
    LOAD_AGAINST_COMMITTED = 32
    'Callables should use committed values as primary/foreign keys during a\n    load.\n    '
    NO_AUTOFLUSH = 64
    ('Loader callables should disable autoflush.',)
    NO_RAISE = 128
    'Loader callables should not raise any assertions'
    DEFERRED_HISTORY_LOAD = 256
    'indicates special load of the previous value of an attribute'
    INCLUDE_PENDING_MUTATIONS = 512
    PASSIVE_OFF = RELATED_OBJECT_OK | NON_PERSISTENT_OK | INIT_OK | CALLABLES_OK | SQL_OK
    'Callables can be emitted in all cases.'
    PASSIVE_RETURN_NO_VALUE = PASSIVE_OFF ^ INIT_OK
    'PASSIVE_OFF ^ INIT_OK'
    PASSIVE_NO_INITIALIZE = PASSIVE_RETURN_NO_VALUE ^ CALLABLES_OK
    'PASSIVE_RETURN_NO_VALUE ^ CALLABLES_OK'
    PASSIVE_NO_FETCH = PASSIVE_OFF ^ SQL_OK
    'PASSIVE_OFF ^ SQL_OK'
    PASSIVE_NO_FETCH_RELATED = PASSIVE_OFF ^ RELATED_OBJECT_OK
    'PASSIVE_OFF ^ RELATED_OBJECT_OK'
    PASSIVE_ONLY_PERSISTENT = PASSIVE_OFF ^ NON_PERSISTENT_OK
    'PASSIVE_OFF ^ NON_PERSISTENT_OK'
    PASSIVE_MERGE = PASSIVE_OFF | NO_RAISE
    'PASSIVE_OFF | NO_RAISE\n\n    Symbol used specifically for session.merge() and similar cases\n\n    '