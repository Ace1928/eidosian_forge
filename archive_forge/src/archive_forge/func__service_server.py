import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import KAMATERA_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kamatera import KamateraNodeDriver
def _service_server(self, method, url, body, headers):
    client_id, secret = (headers['AuthClientId'], headers['AuthSecret'])
    if client_id == 'nosuchuser' and secret == 'nopwd':
        body = self.fixtures.load('failed_auth.json')
        status = httplib.UNAUTHORIZED
    else:
        if url == '/service/server' and json.loads(body).get('ssh-key'):
            body = self.fixtures.load('create_server_sshkey.json')
        else:
            body = self.fixtures.load({'/service/server?datacenter=1': 'datacenters.json', '/service/server?sizes=1&datacenter=EU': 'sizes_datacenter_EU.json', '/service/server?images=1&datacenter=EU': 'images_datacenter_EU.json', '/service/server?capabilities=1&datacenter=EU': 'capabilities_datacenter_EU.json', '/service/server': 'create_server.json'}[url])
        status = httplib.OK
    return (status, body, {}, httplib.responses[status])