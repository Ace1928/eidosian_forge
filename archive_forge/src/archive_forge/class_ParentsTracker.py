import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
class ParentsTracker:
    """
    This is a perf optimization to limit the number of objects we need to visit in tx.replace_all.
    This must be a seperate object so that it is not cloned in apply.
    """

    def __init__(self):
        self.parents: Dict[ParentsTracker, bool] = dict()

    def add(self, parent):
        self.parents[parent] = True

    def recursive_parents(self):
        rv = dict(self.parents)
        worklist = list(self.parents)
        while worklist:
            for parent in worklist.pop().parents:
                if parent not in rv:
                    assert isinstance(parent, ParentsTracker)
                    rv[parent] = True
                    worklist.append(parent)
        return rv.keys()