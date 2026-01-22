from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
def _get_reversed_processed_set(self, uow):
    if not self.prop._reverse_property:
        return None
    process_key = tuple(sorted([self.key] + [p.key for p in self.prop._reverse_property]))
    return uow.memo(('reverse_key', process_key), set)