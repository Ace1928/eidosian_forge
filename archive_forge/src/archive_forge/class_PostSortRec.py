from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
class PostSortRec:
    __slots__ = ('disabled',)

    def __new__(cls, uow, *args):
        key = (cls,) + args
        if key in uow.postsort_actions:
            return uow.postsort_actions[key]
        else:
            uow.postsort_actions[key] = ret = object.__new__(cls)
            ret.disabled = False
            return ret

    def execute_aggregate(self, uow, recs):
        self.execute(uow)