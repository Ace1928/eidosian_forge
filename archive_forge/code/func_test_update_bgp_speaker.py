from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_update_bgp_speaker(self):
    sot = self.operator_cloud.network.update_bgp_speaker(self.SPEAKER.id, advertise_floating_ip_host_routes=False)
    self.assertFalse(sot.advertise_floating_ip_host_routes)