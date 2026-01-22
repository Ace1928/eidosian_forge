from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
@testtools.skip('Under construction')
class ListCommandTestCase(tests.TestCase):

    def setUp(self):
        super(ListCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.list_command = command.ListCommand(self.app, [])