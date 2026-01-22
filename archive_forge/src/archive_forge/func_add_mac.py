import logging
from os_ken.exception import MacAddressDuplicated
from os_ken.lib.mac import haddr_to_str
def add_mac(self, mac, nw_id, nw_id_external=None):
    _nw_id = self.mac_to_net.get(mac)
    if _nw_id == nw_id:
        return
    if _nw_id is None or _nw_id == nw_id_external:
        self.mac_to_net[mac] = nw_id
        LOG.debug('overwrite nw_id: mac %s nw old %s new %s', haddr_to_str(mac), _nw_id, nw_id)
        return
    if nw_id == nw_id_external:
        return
    LOG.warning('duplicated nw_id: mac %s nw old %s new %s', haddr_to_str(mac), _nw_id, nw_id)
    raise MacAddressDuplicated(mac=mac)