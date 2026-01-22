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
def _process_key_switches(self, deplist, uowcommit):
    switchers = self._key_switchers(uowcommit, deplist)
    if switchers:
        for state in uowcommit.session.identity_map.all_states():
            if not issubclass(state.class_, self.parent.class_):
                continue
            dict_ = state.dict
            related = state.get_impl(self.key).get(state, dict_, passive=self._passive_update_flag)
            if related is not attributes.PASSIVE_NO_RESULT and related is not None:
                if self.prop.uselist:
                    if not related:
                        continue
                    related_obj = related[0]
                else:
                    related_obj = related
                related_state = attributes.instance_state(related_obj)
                if related_state in switchers:
                    uowcommit.register_object(state, False, self.passive_updates)
                    sync.populate(related_state, self.mapper, state, self.parent, self.prop.synchronize_pairs, uowcommit, self.passive_updates)