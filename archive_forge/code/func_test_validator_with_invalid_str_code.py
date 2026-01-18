import datetime
import unittest
import pytz
from wsme import utils
def test_validator_with_invalid_str_code(self):
    invalid_str_code = '404'
    self.assertFalse(utils.is_valid_code(invalid_str_code), 'Invalid status code not detected')