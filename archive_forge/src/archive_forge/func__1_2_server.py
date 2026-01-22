import re
import sys
import json
import base64
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import UPCLOUD_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.upcloud import UpcloudDriver, UpcloudResponse
def _1_2_server(self, method, url, body, headers):
    if method == 'POST':
        dbody = json.loads(body)
        storages = dbody['server']['storage_devices']['storage_device']
        if any(['type' in storage and storage['type'] == 'cdrom' for storage in storages]):
            body = self.fixtures.load('api_1_2_server_from_cdrom.json')
        else:
            body = self.fixtures.load('api_1_2_server_from_template.json')
    else:
        body = self.fixtures.load('api_1_2_server.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])