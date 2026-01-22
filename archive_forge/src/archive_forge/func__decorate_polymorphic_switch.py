from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def _decorate_polymorphic_switch(instance_fn, context, query_entity, mapper, result, path, polymorphic_discriminator, adapter, ensure_no_pk):
    if polymorphic_discriminator is not None:
        polymorphic_on = polymorphic_discriminator
    else:
        polymorphic_on = mapper.polymorphic_on
    if polymorphic_on is None:
        return instance_fn
    if adapter:
        polymorphic_on = adapter.columns[polymorphic_on]

    def configure_subclass_mapper(discriminator):
        try:
            sub_mapper = mapper.polymorphic_map[discriminator]
        except KeyError:
            raise AssertionError('No such polymorphic_identity %r is defined' % discriminator)
        else:
            if sub_mapper is mapper:
                return None
            elif not sub_mapper.isa(mapper):
                return False
            return _instance_processor(query_entity, sub_mapper, context, result, path, adapter, _polymorphic_from=mapper)
    polymorphic_instances = util.PopulateDict(configure_subclass_mapper)
    getter = result._getter(polymorphic_on)

    def polymorphic_instance(row):
        discriminator = getter(row)
        if discriminator is not None:
            _instance = polymorphic_instances[discriminator]
            if _instance:
                return _instance(row)
            elif _instance is False:
                identitykey = ensure_no_pk(row)
                if identitykey:
                    raise sa_exc.InvalidRequestError("Row with identity key %s can't be loaded into an object; the polymorphic discriminator column '%s' refers to %s, which is not a sub-mapper of the requested %s" % (identitykey, polymorphic_on, mapper.polymorphic_map[discriminator], mapper))
                else:
                    return None
            else:
                return instance_fn(row)
        else:
            identitykey = ensure_no_pk(row)
            if identitykey:
                raise sa_exc.InvalidRequestError("Row with identity key %s can't be loaded into an object; the polymorphic discriminator column '%s' is NULL" % (identitykey, polymorphic_on))
            else:
                return None
    return polymorphic_instance