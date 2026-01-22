from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_console_get(self, command, expected_response_type):
    server = self._create_server()
    completed_command = command % server.id
    try:
        output = self.nova(completed_command)
        console_type = self._get_column_value_from_single_row_table(output, 'Type')
        self.assertEqual(expected_response_type, console_type, output)
    except exceptions.CommandFailed as cf:
        self.assertIn('HTTP 400', str(cf.stderr))