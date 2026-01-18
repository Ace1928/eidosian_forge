from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_set_bgp_speaker(self):
    arglist = [self._bgp_speaker_name, '--name', 'noob']
    verifylist = [('bgp_speaker', self._bgp_speaker_name), ('name', 'noob')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'noob'}
    self.networkclient.update_bgp_speaker.assert_called_once_with(self._bgp_speaker_name, **attrs)
    self.assertIsNone(result)