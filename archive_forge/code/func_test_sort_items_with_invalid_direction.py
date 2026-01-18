import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_sort_items_with_invalid_direction(self):
    items = self._get_test_items()
    sort_str = 'a:bad_dir'
    self.assertRaises(exceptions.CommandError, utils.sort_items, items, sort_str)