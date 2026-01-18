from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_service_show_nounique(self):
    self.services_mock.find.side_effect = identity_exc.NoUniqueMatch(None)
    arglist = ['nounique_service']
    verifylist = [('service', 'nounique_service')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual("Multiple service matches found for 'nounique_service', use an ID to be more specific.", str(e))