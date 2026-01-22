import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeSize
from libcloud.test.secrets import GRIDSCALE_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gridscale import GridscaleNodeDriver
class GridscaleMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('gridscale')

    def _objects_servers(self, method, url, body, headers):
        body = self.fixtures.load('list_nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_locations(self, method, url, body, headers):
        body = self.fixtures.load('list_locations.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_storages(self, method, url, body, headers):
        body = self.fixtures.load('list_volumes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_storages_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('list_volumes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_ips_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('ex_list_ips.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_ips_56b8d161_325b_4fd4_DELETE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])

    def _objects_storages_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_volumes_empty.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_templates(self, method, url, body, headers):
        body = self.fixtures.load('list_images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_sshkeys(self, method, url, body, headers):
        body = self.fixtures.load('list_key_pairs.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_snapshots(self, method, url, body, headers):
        body = self.fixtures.load('list_volume_snapshots.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f(self, method, url, body, headers):
        if method == 'PATCH':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee(self, method, url, body, headers):
        if method == 'PATCH':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_networks_1196529b_a8de_417f(self, method, url, body, headers):
        if method == 'PATCH':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_servers_POST(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('create_node.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _requests_x123xx1x_123x_1x12_123x_123xxx123x1x_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_node_response_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_node_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_storages_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_volume.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_storages_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_volume_response_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_ips_POST(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('create_ip.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_ips_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_ip_response_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_storages_POST(self, method, url, body, headers):
        body = self.fixtures.load('volume_to_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_ips_POST(self, method, url, body, headers):
        body = self.fixtures.load('ips_to_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_networks_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_network.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_networks_1196529b_a8de_417f_DELETE(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_networks_POST(self, method, url, body, headers):
        body = self.fixtures.load('network_to_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_power_POST(self, method, url, body, headers):
        if method == 'PATCH':
            body = self.fixtures.load('ex_start_node.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_node_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_networks(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_list_networks.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_ips(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_list_ips.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_servers_1479405e_d46c_47a2_91e8_eb43951c899f_DELETE(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_DELETE(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_snapshots_d755de62_4d75_4d61_addd_a5c9743a5deb_DELETE(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise ValueError('Invalid method')

    def _objects_templates_POST(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('create_image.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_snapshots_POST(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('create_image.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_snapshots_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_image_dict.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _objects_templates_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('create_image_dict.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')

    def _objects_templates_12345(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_image.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Invalid method')