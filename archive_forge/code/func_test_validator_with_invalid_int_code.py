import datetime
import unittest
import pytz
from wsme import utils
def test_validator_with_invalid_int_code(self):
    invalid_int_code = 648
    self.assertFalse(utils.is_valid_code(invalid_int_code), 'Invalid status code not detected')