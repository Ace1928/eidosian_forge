import datetime
import unittest
import pytz
from wsme import utils
def test_validator_with_valid_code(self):
    valid_code = 404
    self.assertTrue(utils.is_valid_code(valid_code), 'Valid status code not detected')