import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_with_user_data(self):
    self._check({'foo': 42}, b'I am user data')