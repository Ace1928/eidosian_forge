from breezy import errors, tests
from breezy.tests.per_repository_reference import \
def test_remote_initialize_on_transport_ex(self):
    base = self.make_branch('base')
    trans = self.make_smart_server('stacked')
    repo = self.initialize_and_check_on_transport(base, trans)
    network_name = base.repository._format.network_name()
    self.assertEqual(network_name, repo._format.network_name())