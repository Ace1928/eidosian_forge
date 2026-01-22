from oslotest import base
from aodhclient import utils
class CliQueryToArray(base.BaseTestCase):

    def test_cli_query_to_arrary(self):
        cli_query = 'this<=34;that=string::foo'
        ret_array = utils.cli_to_array(cli_query)
        expected_query = [{'field': 'this', 'type': '', 'value': '34', 'op': 'le'}, {'field': 'that', 'type': 'string', 'value': 'foo', 'op': 'eq'}]
        self.assertEqual(expected_query, ret_array)