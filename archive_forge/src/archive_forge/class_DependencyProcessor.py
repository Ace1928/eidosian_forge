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
class DependencyProcessor:

    def __init__(self, prop):
        self.prop = prop
        self.cascade = prop.cascade
        self.mapper = prop.mapper
        self.parent = prop.parent
        self.secondary = prop.secondary
        self.direction = prop.direction
        self.post_update = prop.post_update
        self.passive_deletes = prop.passive_deletes
        self.passive_updates = prop.passive_updates
        self.enable_typechecks = prop.enable_typechecks
        if self.passive_deletes:
            self._passive_delete_flag = attributes.PASSIVE_NO_INITIALIZE
        else:
            self._passive_delete_flag = attributes.PASSIVE_OFF
        if self.passive_updates:
            self._passive_update_flag = attributes.PASSIVE_NO_INITIALIZE
        else:
            self._passive_update_flag = attributes.PASSIVE_OFF
        self.sort_key = '%s_%s' % (self.parent._sort_key, prop.key)
        self.key = prop.key
        if not self.prop.synchronize_pairs:
            raise sa_exc.ArgumentError("Can't build a DependencyProcessor for relationship %s. No target attributes to populate between parent and child are present" % self.prop)

    @classmethod
    def from_relationship(cls, prop):
        return _direction_to_processor[prop.direction](prop)

    def hasparent(self, state):
        """return True if the given object instance has a parent,
        according to the ``InstrumentedAttribute`` handled by this
        ``DependencyProcessor``.

        """
        return self.parent.class_manager.get_impl(self.key).hasparent(state)

    def per_property_preprocessors(self, uow):
        """establish actions and dependencies related to a flush.

        These actions will operate on all relevant states in
        the aggregate.

        """
        uow.register_preprocessor(self, True)

    def per_property_flush_actions(self, uow):
        after_save = unitofwork.ProcessAll(uow, self, False, True)
        before_delete = unitofwork.ProcessAll(uow, self, True, True)
        parent_saves = unitofwork.SaveUpdateAll(uow, self.parent.primary_base_mapper)
        child_saves = unitofwork.SaveUpdateAll(uow, self.mapper.primary_base_mapper)
        parent_deletes = unitofwork.DeleteAll(uow, self.parent.primary_base_mapper)
        child_deletes = unitofwork.DeleteAll(uow, self.mapper.primary_base_mapper)
        self.per_property_dependencies(uow, parent_saves, child_saves, parent_deletes, child_deletes, after_save, before_delete)

    def per_state_flush_actions(self, uow, states, isdelete):
        """establish actions and dependencies related to a flush.

        These actions will operate on all relevant states
        individually.    This occurs only if there are cycles
        in the 'aggregated' version of events.

        """
        child_base_mapper = self.mapper.primary_base_mapper
        child_saves = unitofwork.SaveUpdateAll(uow, child_base_mapper)
        child_deletes = unitofwork.DeleteAll(uow, child_base_mapper)
        if isdelete:
            before_delete = unitofwork.ProcessAll(uow, self, True, True)
            before_delete.disabled = True
        else:
            after_save = unitofwork.ProcessAll(uow, self, False, True)
            after_save.disabled = True
        if child_saves not in uow.cycles:
            assert child_deletes not in uow.cycles
            child_actions = [(child_saves, False), (child_deletes, True)]
            child_in_cycles = False
        else:
            child_in_cycles = True
        if not isdelete:
            parent_saves = unitofwork.SaveUpdateAll(uow, self.parent.base_mapper)
            parent_deletes = before_delete = None
            if parent_saves in uow.cycles:
                parent_in_cycles = True
        else:
            parent_deletes = unitofwork.DeleteAll(uow, self.parent.base_mapper)
            parent_saves = after_save = None
            if parent_deletes in uow.cycles:
                parent_in_cycles = True
        for state in states:
            sum_ = state.manager[self.key].impl.get_all_pending(state, state.dict, self._passive_delete_flag if isdelete else attributes.PASSIVE_NO_INITIALIZE)
            if not sum_:
                continue
            if isdelete:
                before_delete = unitofwork.ProcessState(uow, self, True, state)
                if parent_in_cycles:
                    parent_deletes = unitofwork.DeleteState(uow, state)
            else:
                after_save = unitofwork.ProcessState(uow, self, False, state)
                if parent_in_cycles:
                    parent_saves = unitofwork.SaveUpdateState(uow, state)
            if child_in_cycles:
                child_actions = []
                for child_state, child in sum_:
                    if child_state not in uow.states:
                        child_action = (None, None)
                    else:
                        deleted, listonly = uow.states[child_state]
                        if deleted:
                            child_action = (unitofwork.DeleteState(uow, child_state), True)
                        else:
                            child_action = (unitofwork.SaveUpdateState(uow, child_state), False)
                    child_actions.append(child_action)
            for child_action, childisdelete in child_actions:
                self.per_state_dependencies(uow, parent_saves, parent_deletes, child_action, after_save, before_delete, isdelete, childisdelete)

    def presort_deletes(self, uowcommit, states):
        return False

    def presort_saves(self, uowcommit, states):
        return False

    def process_deletes(self, uowcommit, states):
        pass

    def process_saves(self, uowcommit, states):
        pass

    def prop_has_changes(self, uowcommit, states, isdelete):
        if not isdelete or self.passive_deletes:
            passive = attributes.PASSIVE_NO_INITIALIZE | attributes.INCLUDE_PENDING_MUTATIONS
        elif self.direction is MANYTOONE:
            passive = attributes.PASSIVE_NO_FETCH_RELATED
        else:
            passive = attributes.PASSIVE_OFF | attributes.INCLUDE_PENDING_MUTATIONS
        for s in states:
            history = uowcommit.get_attribute_history(s, self.key, passive)
            if history and (not history.empty()):
                return True
        else:
            return states and (not self.prop._is_self_referential) and (self.mapper in uowcommit.mappers)

    def _verify_canload(self, state):
        if self.prop.uselist and state is None:
            raise exc.FlushError("Can't flush None value found in collection %s" % (self.prop,))
        elif state is not None and (not self.mapper._canload(state, allow_subtypes=not self.enable_typechecks)):
            if self.mapper._canload(state, allow_subtypes=True):
                raise exc.FlushError('Attempting to flush an item of type %(x)s as a member of collection "%(y)s". Expected an object of type %(z)s or a polymorphic subclass of this type. If %(x)s is a subclass of %(z)s, configure mapper "%(zm)s" to load this subtype polymorphically, or set enable_typechecks=False to allow any subtype to be accepted for flush. ' % {'x': state.class_, 'y': self.prop, 'z': self.mapper.class_, 'zm': self.mapper})
            else:
                raise exc.FlushError('Attempting to flush an item of type %(x)s as a member of collection "%(y)s". Expected an object of type %(z)s or a polymorphic subclass of this type.' % {'x': state.class_, 'y': self.prop, 'z': self.mapper.class_})

    def _synchronize(self, state, child, associationrow, clearkeys, uowcommit):
        raise NotImplementedError()

    def _get_reversed_processed_set(self, uow):
        if not self.prop._reverse_property:
            return None
        process_key = tuple(sorted([self.key] + [p.key for p in self.prop._reverse_property]))
        return uow.memo(('reverse_key', process_key), set)

    def _post_update(self, state, uowcommit, related, is_m2o_delete=False):
        for x in related:
            if not is_m2o_delete or x is not None:
                uowcommit.register_post_update(state, [r for l, r in self.prop.synchronize_pairs])
                break

    def _pks_changed(self, uowcommit, state):
        raise NotImplementedError()

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.prop)