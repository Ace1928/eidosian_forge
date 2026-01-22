import configparser as config_parser
import os
from tempest.lib.cli import base
def assertTableHeaders(self, field_names, table_headers):
    """Assert that field_names and table_headers are equal.

        :param field_names: field names from the output table of the cmd
        :param table_headers: table headers output from cmd
        """
    self.assertEqual(sorted(field_names), sorted(table_headers))