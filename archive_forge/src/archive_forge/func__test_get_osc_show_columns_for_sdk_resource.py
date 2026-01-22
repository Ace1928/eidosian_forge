import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def _test_get_osc_show_columns_for_sdk_resource(self, sdk_resource, column_map, expected_display_columns, expected_attr_columns):
    display_columns, attr_columns = utils.get_osc_show_columns_for_sdk_resource(sdk_resource, column_map)
    self.assertEqual(expected_display_columns, display_columns)
    self.assertEqual(expected_attr_columns, attr_columns)