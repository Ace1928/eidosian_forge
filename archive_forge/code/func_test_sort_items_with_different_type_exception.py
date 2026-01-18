import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_sort_items_with_different_type_exception(self):
    item1 = {'a': 2}
    item2 = {'a': 3}
    item3 = {'a': None}
    item4 = {'a': 1}
    items = [item1, item2, item3, item4]
    sort_str = 'a'
    self.assertRaises(TypeError, utils.sort_items, items, sort_str)