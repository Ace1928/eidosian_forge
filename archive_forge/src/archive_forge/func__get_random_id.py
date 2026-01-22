import random
import string
from tests.compat import unittest, mock
import boto
def _get_random_id(self, length=14):
    return ''.join([random.choice(string.ascii_letters) for i in range(length)])