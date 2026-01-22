from os_ken.services.protocols.bgp.signals import SignalBus
def adj_rib_in_changed(self, peer, received_route):
    return self.emit_signal(self.BGP_ADJ_RIB_IN_CHANGED, {'peer': peer, 'received_route': received_route})