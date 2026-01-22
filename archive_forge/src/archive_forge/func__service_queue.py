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
def _service_queue(self, method, url, body, headers):
    if not hasattr(self, '_service_queue_call_count'):
        self._service_queue_call_count = 0
    self._service_queue_call_count += 1
    body = self.fixtures.load({'/service/queue?id=12345': 'queue_12345-%s.json' % self._service_queue_call_count}[url])
    status = httplib.OK
    return (status, body, {}, httplib.responses[status])