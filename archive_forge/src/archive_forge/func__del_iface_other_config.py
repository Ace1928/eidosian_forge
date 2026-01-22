import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _del_iface_other_config(tables, *_):
    row = fn(tables)
    if not row:
        return None
    other_config = row.other_config
    if key in other_config:
        other_config.pop(key)
        row.other_config = other_config