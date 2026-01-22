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
@util.memoized_property
def _mapper_for_dep(self):
    """return a dynamic mapping of (Mapper, DependencyProcessor) to
        True or False, indicating if the DependencyProcessor operates
        on objects of that Mapper.

        The result is stored in the dictionary persistently once
        calculated.

        """
    return util.PopulateDict(lambda tup: tup[0]._props.get(tup[1].key) is tup[1].prop)