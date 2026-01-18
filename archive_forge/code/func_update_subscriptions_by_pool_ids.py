from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def update_subscriptions_by_pool_ids(self, pool_ids):
    changed = False
    consumed_pools = RhsmPools(self.module, consumed=True)
    existing_pools = {}
    serials_to_remove = []
    for p in consumed_pools:
        pool_id = p.get_pool_id()
        quantity_used = p.get_quantity_used()
        existing_pools[pool_id] = quantity_used
        quantity = pool_ids.get(pool_id, 0)
        if quantity is not None and quantity != quantity_used:
            serials_to_remove.append(p.Serial)
    serials = self.unsubscribe(serials=serials_to_remove)
    missing_pools = {}
    for pool_id, quantity in sorted(pool_ids.items()):
        quantity_used = existing_pools.get(pool_id, 0)
        if quantity is None and quantity_used == 0 or quantity not in (None, 0, quantity_used):
            missing_pools[pool_id] = quantity
    self.subscribe_by_pool_ids(missing_pools)
    if missing_pools or serials:
        changed = True
    return {'changed': changed, 'subscribed_pool_ids': list(missing_pools.keys()), 'unsubscribed_serials': serials}