import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_safe_sort_key_mapping_ne(self):
    list1 = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
    data1 = self._create_dict_from_list(list1)
    list2 = [('yellow', 3), ('blue', 4), ('yellow', 1), ('blue', 2), ('red', 1)]
    data2 = self._create_dict_from_list(list2)
    self.assertNotEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))