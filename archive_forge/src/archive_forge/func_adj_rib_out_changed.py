from os_ken.services.protocols.bgp.signals import SignalBus
def adj_rib_out_changed(self, peer, sent_route):
    return self.emit_signal(self.BGP_ADJ_RIB_OUT_CHANGED, {'peer': peer, 'sent_route': sent_route})