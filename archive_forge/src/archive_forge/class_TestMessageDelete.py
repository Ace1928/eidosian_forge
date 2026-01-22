from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestMessageDelete(TestMessage):

    def setUp(self):
        super(TestMessageDelete, self).setUp()
        self.message = manila_fakes.FakeMessage.create_one_message()
        self.messages_mock.get.return_value = self.message
        self.cmd = osc_messages.DeleteMessage(self.app, None)

    def test_message_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_message_delete(self):
        arglist = [self.message.id]
        verifylist = [('message', [self.message.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.messages_mock.delete.assert_called_with(self.message)
        self.assertIsNone(result)

    def test_message_delete_multiple(self):
        messages = manila_fakes.FakeMessage.create_messages(count=2)
        arglist = [messages[0].id, messages[1].id]
        verifylist = [('message', [messages[0].id, messages[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.messages_mock.delete.call_count, len(messages))
        self.assertIsNone(result)

    def test_message_delete_exception(self):
        arglist = [self.message.id]
        verifylist = [('message', [self.message.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.messages_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)