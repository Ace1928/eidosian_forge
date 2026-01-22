import ddt
from manilaclient.tests.functional import base
@ddt.ddt
class MessagesReadOnlyTest(base.BaseTestCase):

    @ddt.data(('admin', '2.37'), ('user', '2.37'))
    @ddt.unpack
    def test_message_list(self, role, microversion):
        self.skip_if_microversion_not_supported(microversion)
        self.clients[role].manila('message-list', microversion=microversion)