from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def _get_table_headers(self, raw_output):
    table = self.parser.table(raw_output)
    return table['headers']