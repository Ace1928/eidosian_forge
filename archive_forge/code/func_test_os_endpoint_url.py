import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_os_endpoint_url(self):
    cl = get_authed_endpoint_url()
    self.assertEqual('volume/v100', cl.os_endpoint)
    self.assertEqual('volume/v100', cl.management_url)