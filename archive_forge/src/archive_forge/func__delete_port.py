import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _delete_port(tables, *_):
    bridge = _get_bridge(tables, bridge_name)
    if not bridge:
        return
    port = fn(tables)
    if not port:
        return
    ports = bridge.ports
    ports.remove(port)
    bridge.ports = ports