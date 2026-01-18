import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_with_network_data(self):
    self._check({'foo': 42}, network_data={'networks': {}})