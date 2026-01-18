import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_with_user_data_as_string(self):
    self._check({'foo': 42}, u'I am user data')