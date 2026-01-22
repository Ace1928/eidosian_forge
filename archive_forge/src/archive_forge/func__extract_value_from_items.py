import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _extract_value_from_items(self, key, items):
    for d in items:
        for k, v in d.items():
            if k == key:
                return v