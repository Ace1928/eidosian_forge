from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
class TestCompletionAlternatives(base.TestBase):

    def given_cmdo_data(self):
        cmdo = 'image server'
        data = [('image', 'create'), ('image_create', '--eolus'), ('server', 'meta ssh'), ('server_meta_delete', '--wilson'), ('server_ssh', '--sunlight')]
        return (cmdo, data)

    def then_data(self, content):
        self.assertIn("  cmds='image server'\n", content)
        self.assertIn("  cmds_image='create'\n", content)
        self.assertIn("  cmds_image_create='--eolus'\n", content)
        self.assertIn("  cmds_server='meta ssh'\n", content)
        self.assertIn("  cmds_server_meta_delete='--wilson'\n", content)
        self.assertIn("  cmds_server_ssh='--sunlight'\n", content)

    def test_complete_no_code(self):
        output = FakeStdout()
        sot = complete.CompleteNoCode('doesNotMatter', output)
        sot.write(*self.given_cmdo_data())
        self.then_data(output.content)

    def test_complete_bash(self):
        output = FakeStdout()
        sot = complete.CompleteBash('openstack', output)
        sot.write(*self.given_cmdo_data())
        self.then_data(output.content)
        self.assertIn('_openstack()\n', output.content[0])
        self.assertIn('complete -F _openstack openstack\n', output.content[-1])

    def test_complete_command_parser(self):
        sot = complete.CompleteCommand(mock.Mock(), mock.Mock())
        parser = sot.get_parser('nothing')
        self.assertEqual('nothing', parser.prog)
        self.assertEqual('print bash completion command\n    ', parser.description)