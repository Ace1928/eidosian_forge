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
class VPSNetMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('vpsnet')

    def _nodes_api10json_sizes(self, method, url, body, headers):
        body = '[{"slice":{"virtual_machine_id":8592,"id":12256,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12258,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12434,"consumer_id":0}}]'
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _nodes_api10json_create(self, method, url, body, headers):
        body = '[{"slice":{"virtual_machine_id":8592,"id":12256,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12258,"consumer_id":0}},\n                   {"slice":{"virtual_machine_id":null,"id":12434,"consumer_id":0}}]'
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _virtual_machines_2222_api10json_delete_fail(self, method, url, body, headers):
        return (httplib.FORBIDDEN, '', {}, httplib.responses[httplib.FORBIDDEN])

    def _virtual_machines_2222_api10json_delete(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _virtual_machines_1384_reboot_api10json_reboot(self, method, url, body, headers):
        body = '{\n              "virtual_machine":\n                {\n                  "running": true,\n                  "updated_at": "2009-05-15T06:55:02-04:00",\n                  "power_action_pending": false,\n                  "system_template_id": 41,\n                  "id": 1384,\n                  "cloud_id": 3,\n                  "domain_name": "demodomain.com",\n                  "hostname": "web01",\n                  "consumer_id": 0,\n                  "backups_enabled": false,\n                  "password": "a8hjsjnbs91",\n                  "label": "foo",\n                  "slices_count": null,\n                  "created_at": "2009-04-16T08:17:39-04:00"\n                }\n              }'
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _virtual_machines_api10json_create(self, method, url, body, headers):
        body = '{\n              "virtual_machine":\n                {\n                  "running": true,\n                  "updated_at": "2009-05-15T06:55:02-04:00",\n                  "power_action_pending": false,\n                  "system_template_id": 41,\n                  "id": 1384,\n                  "cloud_id": 3,\n                  "domain_name": "demodomain.com",\n                  "hostname": "web01",\n                  "consumer_id": 0,\n                  "backups_enabled": false,\n                  "password": "a8hjsjnbs91",\n                  "label": "foo",\n                  "slices_count": null,\n                  "created_at": "2009-04-16T08:17:39-04:00"\n                }\n              }'
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _virtual_machines_api10json_virtual_machines(self, method, url, body, headers):
        body = '     [{\n              "virtual_machine":\n                {\n                  "running": true,\n                  "updated_at": "2009-05-15T06:55:02-04:00",\n                  "power_action_pending": false,\n                  "system_template_id": 41,\n                  "id": 1384,\n                  "cloud_id": 3,\n                  "domain_name": "demodomain.com",\n                  "hostname": "web01",\n                  "consumer_id": 0,\n                  "backups_enabled": false,\n                  "password": "a8hjsjnbs91",\n                  "label": "Web Server 01",\n                  "slices_count": null,\n                  "created_at": "2009-04-16T08:17:39-04:00"\n                }\n              },\n              {\n                "virtual_machine":\n                  {\n                    "running": true,\n                    "updated_at": "2009-05-15T06:55:02-04:00",\n                    "power_action_pending": false,\n                    "system_template_id": 41,\n                    "id": 1385,\n                    "cloud_id": 3,\n                    "domain_name": "demodomain.com",\n                    "hostname": "mysql01",\n                    "consumer_id": 0,\n                    "backups_enabled": false,\n                    "password": "dsi8h38hd2s",\n                    "label": "MySQL Server 01",\n                    "slices_count": null,\n                    "created_at": "2009-04-16T08:17:39-04:00"\n                  }\n                }]'
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _available_clouds_api10json_templates(self, method, url, body, headers):
        body = self.fixtures.load('_available_clouds_api10json_templates.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _available_clouds_api10json_create(self, method, url, body, headers):
        body = '\n        [{"cloud":{"system_templates":[{"id":9,"label":"Ubuntu 8.04 x64"}],"id":2,"label":"USA VPS Cloud"}}]\n        '
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])