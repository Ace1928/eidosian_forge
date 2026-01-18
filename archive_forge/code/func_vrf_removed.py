from os_ken.services.protocols.bgp.signals import SignalBus
def vrf_removed(self, route_dist):
    return self.emit_signal(self.BGP_VRF_REMOVED, route_dist)