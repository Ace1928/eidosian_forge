from unittest import mock
from saharaclient.osc import utils
from saharaclient.tests.unit import base
def test_prepare_column_headers(self):
    columns1 = ['first', 'second_column']
    self.assertEqual(['First', 'Second column'], utils.prepare_column_headers(columns1))
    columns2 = ['First', 'Second column']
    self.assertEqual(['First', 'Second column'], utils.prepare_column_headers(columns2))
    columns3 = ['first', 'second_column']
    self.assertEqual(['First', 'Second'], utils.prepare_column_headers(columns3, remap={'second_column': 'second'}))