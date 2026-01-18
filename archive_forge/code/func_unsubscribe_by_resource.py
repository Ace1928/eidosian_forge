import collections
from oslo_log import log as logging
from oslo_utils import reflection
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import priority_group
from neutron_lib.db import utils as db_utils
def unsubscribe_by_resource(self, callback, resource):
    """Unsubscribe callback for any event associated to the resource.

        :param callback: the callback.
        :param resource: the resource.
        """
    callback_id = self._find(callback)
    if callback_id:
        if resource in self._index[callback_id]:
            for event in self._index[callback_id][resource]:
                self._del_callback(self._callbacks[resource][event], callback_id)
            del self._index[callback_id][resource]
            if not self._index[callback_id]:
                del self._index[callback_id]