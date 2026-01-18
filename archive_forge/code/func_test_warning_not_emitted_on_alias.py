import warnings
import os_service_types
from os_service_types import exc
from os_service_types.tests import base
def test_warning_not_emitted_on_alias(self):
    with warnings.catch_warnings(record=True) as w:
        self.service_types.get_service_type('volumev2')
        self.assertEqual(0, len(w))