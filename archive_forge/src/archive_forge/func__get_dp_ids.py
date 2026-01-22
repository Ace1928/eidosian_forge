import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _get_dp_ids(tables):
    bridges = tables.get('Bridge')
    if not bridges:
        return None
    for bridge in bridges.rows.values():
        datapath_ids = [dpidlib.str_to_dpid(dp_id) for dp_id in bridge.datapath_id]
        if datapath_id in datapath_ids:
            openvswitch = tables['Open_vSwitch'].rows
            if openvswitch:
                row = openvswitch.get(list(openvswitch.keys())[0])
                return row.external_ids.get('system-id')
    return None