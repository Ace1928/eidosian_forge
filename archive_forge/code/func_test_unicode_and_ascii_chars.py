import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_unicode_and_ascii_chars(self):
    source = random_string(10, UNICODE_CHARS) + random_string(10) + random_string(10, self.DIFFICULT_CHARS)
    password = random_string(20, source)
    username = random_string(20, source)
    service = random_string(20, source)
    self.check_set_get(service, username, password)