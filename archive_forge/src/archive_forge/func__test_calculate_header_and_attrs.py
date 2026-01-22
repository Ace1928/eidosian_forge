import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def _test_calculate_header_and_attrs(self, parsed_args_columns, expected_headers, expected_attrs):
    column_headers = ('ID', 'Name', 'Fixed IP Addresses')
    columns = ('id', 'name', 'fixed_ips')
    parsed_args = mock.Mock()
    parsed_args.columns = parsed_args_columns
    ret_headers, ret_attrs = utils.calculate_header_and_attrs(column_headers, columns, parsed_args)
    self.assertEqual(expected_headers, ret_headers)
    self.assertEqual(expected_attrs, ret_attrs)
    if parsed_args_columns:
        self.assertEqual(expected_headers, parsed_args.columns)
    else:
        self.assertFalse(parsed_args.columns)