import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VPSNET_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vpsnet import VPSNetNodeDriver
def _virtual_machines_api10json_virtual_machines(self, method, url, body, headers):
    body = '     [{\n              "virtual_machine":\n                {\n                  "running": true,\n                  "updated_at": "2009-05-15T06:55:02-04:00",\n                  "power_action_pending": false,\n                  "system_template_id": 41,\n                  "id": 1384,\n                  "cloud_id": 3,\n                  "domain_name": "demodomain.com",\n                  "hostname": "web01",\n                  "consumer_id": 0,\n                  "backups_enabled": false,\n                  "password": "a8hjsjnbs91",\n                  "label": "Web Server 01",\n                  "slices_count": null,\n                  "created_at": "2009-04-16T08:17:39-04:00"\n                }\n              },\n              {\n                "virtual_machine":\n                  {\n                    "running": true,\n                    "updated_at": "2009-05-15T06:55:02-04:00",\n                    "power_action_pending": false,\n                    "system_template_id": 41,\n                    "id": 1385,\n                    "cloud_id": 3,\n                    "domain_name": "demodomain.com",\n                    "hostname": "mysql01",\n                    "consumer_id": 0,\n                    "backups_enabled": false,\n                    "password": "dsi8h38hd2s",\n                    "label": "MySQL Server 01",\n                    "slices_count": null,\n                    "created_at": "2009-04-16T08:17:39-04:00"\n                  }\n                }]'
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])