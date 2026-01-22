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
class DeleteAll(PostSortRec):
    __slots__ = ('mapper', 'sort_key')

    def __init__(self, uow, mapper):
        self.mapper = mapper
        self.sort_key = ('DeleteAll', mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        util.preloaded.orm_persistence.delete_obj(self.mapper, uow.states_for_mapper_hierarchy(self.mapper, True, False), uow)

    def per_state_flush_actions(self, uow):
        states = list(uow.states_for_mapper_hierarchy(self.mapper, True, False))
        base_mapper = self.mapper.base_mapper
        save_all = SaveUpdateAll(uow, base_mapper)
        for state in states:
            action = DeleteState(uow, state)
            uow.dependencies.add((save_all, action))
            yield action
        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, True)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.mapper)