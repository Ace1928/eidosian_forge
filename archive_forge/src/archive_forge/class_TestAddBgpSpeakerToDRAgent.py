from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_dragent
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestAddBgpSpeakerToDRAgent(fakes.TestNeutronDynamicRoutingOSCV2):
    _bgp_speaker = fakes.FakeBgpSpeaker.create_one_bgp_speaker()
    _bgp_dragent = fakes.FakeDRAgent.create_one_dragent()
    _bgp_speaker_id = _bgp_speaker['id']
    _bgp_dragent_id = _bgp_dragent['id']

    def setUp(self):
        super(TestAddBgpSpeakerToDRAgent, self).setUp()
        self.cmd = bgp_dragent.AddBgpSpeakerToDRAgent(self.app, self.namespace)

    def test_add_bgp_speaker_to_dragent(self):
        arglist = [self._bgp_dragent_id, self._bgp_speaker_id]
        verifylist = [('dragent_id', self._bgp_dragent_id), ('bgp_speaker', self._bgp_speaker_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.networkclient, 'add_bgp_speaker_to_dragent', return_value=None):
            result = self.cmd.take_action(parsed_args)
            self.networkclient.add_bgp_speaker_to_dragent.assert_called_once_with(self._bgp_dragent_id, self._bgp_speaker_id)
            self.assertIsNone(result)