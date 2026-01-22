import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _match_row(tables):
    return next((r for r in tables[table].rows.values() if fn(r)), None)