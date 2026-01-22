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
def _mappers(self, uow):
    if self.fromparent:
        return iter((m for m in self.dependency_processor.parent.self_and_descendants if uow._mapper_for_dep[m, self.dependency_processor]))
    else:
        return self.dependency_processor.mapper.self_and_descendants