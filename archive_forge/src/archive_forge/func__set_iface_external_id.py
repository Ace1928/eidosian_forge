import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _set_iface_external_id(tables, *_):
    row = fn(tables)
    if not row:
        return None
    external_ids = row.external_ids
    external_ids[key] = val
    row.external_ids = external_ids