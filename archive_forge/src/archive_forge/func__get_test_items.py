import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def _get_test_items(self):
    item1 = {'a': 1, 'b': 2}
    item2 = {'a': 1, 'b': 3}
    item3 = {'a': 2, 'b': 2}
    item4 = {'a': 2, 'b': 1}
    return [item1, item2, item3, item4]