import sys
import unittest
from unittest.mock import patch
from libcloud.http import LibcloudConnection
from libcloud.test import no_internet
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import OVH_PARAMS
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ovh import OvhNodeDriver
from libcloud.test.common.test_ovh import BaseOvhMockHttp
class OvhMockHttp(BaseOvhMockHttp):
    """Fixtures needed for tests related to rating model"""
    fixtures = ComputeFileFixtures('ovh')

    def _json_1_0_auth_time_get(self, method, url, body, headers):
        body = self.fixtures.load('auth_time_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_region_get(self, method, url, body, headers):
        body = self.fixtures.load('region_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_flavor_get(self, method, url, body, headers):
        body = self.fixtures.load('flavor_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_flavor_region_SBG1_get(self, method, url, body, headers):
        body = self.fixtures.load('flavor_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_flavor_foo_id_get(self, method, url, body, headers):
        body = self.fixtures.load('flavor_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_image_get(self, method, url, body, headers):
        body = self.fixtures.load('image_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_image_foo_id_get(self, method, url, body, headers):
        body = self.fixtures.load('image_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_sshkey_region_SBG1_get(self, method, url, body, headers):
        body = self.fixtures.load('ssh_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_sshkey_post(self, method, url, body, headers):
        body = self.fixtures.load('ssh_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_ssh_mykey_get(self, method, url, body, headers):
        body = self.fixtures.load('ssh_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_instance_get(self, method, url, body, headers):
        body = self.fixtures.load('instance_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_instance_foo_get(self, method, url, body, headers):
        body = self.fixtures.load('instance_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_instance_foo_delete(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_instance_post(self, method, url, body, headers):
        body = self.fixtures.load('instance_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_get(self, method, url, body, headers):
        body = self.fixtures.load('volume_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_post(self, method, url, body, headers):
        body = self.fixtures.load('volume_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_foo_get(self, method, url, body, headers):
        body = self.fixtures.load('volume_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_foo_delete(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_foo_attach_post(self, method, url, body, headers):
        body = self.fixtures.load('volume_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_foo_detach_post(self, method, url, body, headers):
        body = self.fixtures.load('volume_get_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_snapshot_region_SBG_1_get(self, method, url, body, headers):
        body = self.fixtures.load('volume_snapshot_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_snapshot_get(self, method, url, body, headers):
        body = self.fixtures.load('volume_snapshot_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_snapshot_foo_get(self, method, url, body, headers):
        body = self.fixtures.load('volume_snapshot_get_details.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_snapshot_foo_snap_delete(self, method, url, body, headers):
        return (httplib.OK, None, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_volume_foo_snapshot__post(self, method, url, body, headers):
        body = self.fixtures.load('volume_snapshot_get_details.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_subsidiaryPrice_ovhSubsidiary_US_flavorId_foo_id_get(self, method, url, body, headers):
        return self._json_1_0_cloud_subsidiaryPrice_flavorId_foo_id_ovhSubsidiary_US_get(method, url, body, headers)

    def _json_1_0_cloud_subsidiaryPrice_flavorId_foo_id_ovhSubsidiary_US_get(self, method, url, body, headers):
        body = self.fixtures.load('pricing_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_1_0_cloud_project_project_id_instance_get_invalid_app_key_error(self, method, url, body, headers):
        body = '{"message":"Invalid application key"}'
        return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.OK])