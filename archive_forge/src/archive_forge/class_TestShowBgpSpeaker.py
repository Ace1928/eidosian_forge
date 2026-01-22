from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestShowBgpSpeaker(fakes.TestNeutronDynamicRoutingOSCV2):
    _one_bgp_speaker = fakes.FakeBgpSpeaker.create_one_bgp_speaker()
    data = (_one_bgp_speaker['advertise_floating_ip_host_routes'], _one_bgp_speaker['advertise_tenant_networks'], _one_bgp_speaker['id'], _one_bgp_speaker['ip_version'], _one_bgp_speaker['local_as'], _one_bgp_speaker['name'], _one_bgp_speaker['networks'], _one_bgp_speaker['peers'], _one_bgp_speaker['tenant_id'])
    _bgp_speaker = _one_bgp_speaker
    _bgp_speaker_name = _one_bgp_speaker['name']
    columns = ('advertise_floating_ip_host_routes', 'advertise_tenant_networks', 'id', 'ip_version', 'local_as', 'name', 'networks', 'peers', 'project_id')

    def setUp(self):
        super(TestShowBgpSpeaker, self).setUp()
        self.networkclient.get_bgp_speaker = mock.Mock(return_value=self._bgp_speaker)
        self.cmd = bgp_speaker.ShowBgpSpeaker(self.app, self.namespace)

    def test_bgp_speaker_show(self):
        arglist = [self._bgp_speaker_name]
        verifylist = [('bgp_speaker', self._bgp_speaker_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        data = self.cmd.take_action(parsed_args)
        self.networkclient.get_bgp_speaker.assert_called_once_with(self._bgp_speaker_name)
        self.assertEqual(self.columns, data[0])
        self.assertEqual(self.data, data[1])