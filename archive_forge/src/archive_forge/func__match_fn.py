import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _match_fn(row):
    row_dpid = dpidlib.str_to_dpid(str(row.datapath_id[0]))
    return row_dpid == datapath_id