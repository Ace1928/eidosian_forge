import datetime
from novaclient.tests.functional import base
def _get_num_servers_from_usage_output(self):
    output = self.nova('usage')
    servers = self._get_column_value_from_single_row_table(output, 'Servers')
    return int(servers)