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
class ManyToOneDP(DependencyProcessor):

    def __init__(self, prop):
        DependencyProcessor.__init__(self, prop)
        for mapper in self.mapper.self_and_descendants:
            mapper._dependency_processors.append(DetectKeySwitch(prop))

    def per_property_dependencies(self, uow, parent_saves, child_saves, parent_deletes, child_deletes, after_save, before_delete):
        if self.post_update:
            parent_post_updates = unitofwork.PostUpdateAll(uow, self.parent.primary_base_mapper, False)
            parent_pre_updates = unitofwork.PostUpdateAll(uow, self.parent.primary_base_mapper, True)
            uow.dependencies.update([(child_saves, after_save), (parent_saves, after_save), (after_save, parent_post_updates), (after_save, parent_pre_updates), (before_delete, parent_pre_updates), (parent_pre_updates, child_deletes), (parent_pre_updates, parent_deletes)])
        else:
            uow.dependencies.update([(child_saves, after_save), (after_save, parent_saves), (parent_saves, child_deletes), (parent_deletes, child_deletes)])

    def per_state_dependencies(self, uow, save_parent, delete_parent, child_action, after_save, before_delete, isdelete, childisdelete):
        if self.post_update:
            if not isdelete:
                parent_post_updates = unitofwork.PostUpdateAll(uow, self.parent.primary_base_mapper, False)
                if childisdelete:
                    uow.dependencies.update([(after_save, parent_post_updates), (parent_post_updates, child_action)])
                else:
                    uow.dependencies.update([(save_parent, after_save), (child_action, after_save), (after_save, parent_post_updates)])
            else:
                parent_pre_updates = unitofwork.PostUpdateAll(uow, self.parent.primary_base_mapper, True)
                uow.dependencies.update([(before_delete, parent_pre_updates), (parent_pre_updates, delete_parent), (parent_pre_updates, child_action)])
        elif not isdelete:
            if not childisdelete:
                uow.dependencies.update([(child_action, after_save), (after_save, save_parent)])
            else:
                uow.dependencies.update([(after_save, save_parent)])
        elif childisdelete:
            uow.dependencies.update([(delete_parent, child_action)])

    def presort_deletes(self, uowcommit, states):
        if self.cascade.delete or self.cascade.delete_orphan:
            for state in states:
                history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
                if history:
                    if self.cascade.delete_orphan:
                        todelete = history.sum()
                    else:
                        todelete = history.non_deleted()
                    for child in todelete:
                        if child is None:
                            continue
                        uowcommit.register_object(child, isdelete=True, operation='delete', prop=self.prop)
                        t = self.mapper.cascade_iterator('delete', child)
                        for c, m, st_, dct_ in t:
                            uowcommit.register_object(st_, isdelete=True)

    def presort_saves(self, uowcommit, states):
        for state in states:
            uowcommit.register_object(state, operation='add', prop=self.prop)
            if self.cascade.delete_orphan:
                history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
                if history:
                    for child in history.deleted:
                        if self.hasparent(child) is False:
                            uowcommit.register_object(child, isdelete=True, operation='delete', prop=self.prop)
                            t = self.mapper.cascade_iterator('delete', child)
                            for c, m, st_, dct_ in t:
                                uowcommit.register_object(st_, isdelete=True)

    def process_deletes(self, uowcommit, states):
        if self.post_update and (not self.cascade.delete_orphan) and (not self.passive_deletes == 'all'):
            for state in states:
                self._synchronize(state, None, None, True, uowcommit)
                if state and self.post_update:
                    history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
                    if history:
                        self._post_update(state, uowcommit, history.sum(), is_m2o_delete=True)

    def process_saves(self, uowcommit, states):
        for state in states:
            history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_NO_INITIALIZE)
            if history:
                if history.added:
                    for child in history.added:
                        self._synchronize(state, child, None, False, uowcommit, 'add')
                elif history.deleted:
                    self._synchronize(state, None, None, True, uowcommit, 'delete')
                if self.post_update:
                    self._post_update(state, uowcommit, history.sum())

    def _synchronize(self, state, child, associationrow, clearkeys, uowcommit, operation=None):
        if state is None or (not self.post_update and uowcommit.is_deleted(state)):
            return
        if operation is not None and child is not None and (not uowcommit.session._contains_state(child)):
            util.warn("Object of type %s not in session, %s operation along '%s' won't proceed" % (mapperutil.state_class_str(child), operation, self.prop))
            return
        if clearkeys or child is None:
            sync.clear(state, self.parent, self.prop.synchronize_pairs)
        else:
            self._verify_canload(child)
            sync.populate(child, self.mapper, state, self.parent, self.prop.synchronize_pairs, uowcommit, False)