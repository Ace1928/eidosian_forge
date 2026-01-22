from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
@testtools.skip('Under construction')
class DeleteCommandTestCase(tests.TestCase):

    def setUp(self):
        super(DeleteCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.delete_command = command.DeleteCommand(self.app, [])