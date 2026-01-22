import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
class AbiquoMockHttp(MockHttp):
    """
    Mock the functionality of the remote Abiquo API.
    """
    fixtures = ComputeFileFixtures('abiquo')
    fixture_tag = 'default'

    def _api_login(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic c29uOmdvdGVu':
            expected_response = self.fixtures.load('unauthorized_user.html')
            expected_status = httplib.UNAUTHORIZED
        else:
            expected_response = self.fixtures.load('login.xml')
            expected_status = httplib.OK
        return (expected_status, expected_response, {}, '')

    def _api_cloud_virtualdatacenters(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('vdcs.xml'), {}, '')

    def _api_cloud_virtualdatacenters_4(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('vdc_4.xml'), {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances(self, method, url, body, headers):
        if method == 'POST':
            vapp_name = ET.XML(body).findtext('name')
            if vapp_name == 'libcloud_test_group':
                response = self.fixtures.load('vdc_4_vapp_creation_ok.xml')
                return (httplib.OK, response, {}, '')
            elif vapp_name == 'new_group_name':
                response = self.fixtures.load('vdc_4_vapp_creation_ok.xml')
                return (httplib.OK, response, {}, '')
        else:
            return (httplib.OK, self.fixtures.load('vdc_4_vapps.xml'), {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_5(self, method, url, body, headers):
        if method == 'GET':
            if headers['Authorization'] == 'Basic dmU6Z2V0YQ==':
                response = self.fixtures.load('vdc_4_vapp_5_needs_sync.xml')
            else:
                response = self.fixtures.load('vdc_4_vapp_5.xml')
            return (httplib.OK, response, {}, '')
        else:
            return (httplib.NO_CONTENT, '', {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6(self, method, url, body, headers):
        if method == 'GET':
            response = self.fixtures.load('vdc_4_vapp_6.xml')
            return (httplib.OK, response, {}, '')
        else:
            return (httplib.NO_CONTENT, '', {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_tasks_1da8c8b6_86f6_49ef_9d29_57dcc73b875a(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic bXV0ZW46cm9zaGk=':
            response = self.fixtures.load('vdc_4_vapp_6_undeploy_task_failed.xml')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_undeploy_task.xml')
        return (httplib.OK, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_5_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            response = self.fixtures.load('vdc_4_vapp_5_vms.xml')
            return (httplib.OK, response, {}, '')
        elif method == 'POST':
            response = self.fixtures.load('vdc_4_vapp_6_vm_creation_ok.xml')
            return (httplib.CREATED, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            if headers['Authorization'] == 'Basic dmU6Z2V0YQ==':
                response = self.fixtures.load('vdc_4_vapp_6_vms_allocated.xml')
            else:
                response = self.fixtures.load('vdc_4_vapp_6_vms.xml')
            return (httplib.OK, response, {}, '')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_vm_creation_ok.xml')
            return (httplib.CREATED, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic Z286dHJ1bmtz' or headers['Authorization'] == 'Basic bXV0ZW46cm9zaGk=':
            response = self.fixtures.load('vdc_4_vapp_6_vm_3_deployed.xml')
        elif headers['Authorization'] == 'Basic dmU6Z2V0YQ==':
            response = self.fixtures.load('vdc_4_vapp_6_vm_3_allocated.xml')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_vm_3.xml')
        return (httplib.OK, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_action_deploy(self, method, url, body, headers):
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_deploy.xml')
        return (httplib.CREATED, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_tasks_b44fe278_6b0f_4dfb_be81_7c03006a93cb(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic dGVuOnNoaW4=':
            response = self.fixtures.load('vdc_4_vapp_6_vm_3_deploy_task_failed.xml')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_vm_3_deploy_task.xml')
        return (httplib.OK, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_action_undeploy(self, method, url, body, headers):
        response = self.fixtures.load('vdc_4_vapp_6_undeploy.xml')
        return (httplib.OK, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_action_reset(self, method, url, body, headers):
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_reset.xml')
        return (httplib.CREATED, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_tasks_a8c9818e_f389_45b7_be2c_3db3a9689940(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic bXV0ZW46cm9zaGk=':
            response = self.fixtures.load('vdc_4_vapp_6_undeploy_task_failed.xml')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_vm_3_reset_task.xml')
        return (httplib.OK, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_action_undeploy(self, method, url, body, headers):
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_undeploy.xml')
        return (httplib.CREATED, response, {}, '')

    def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_network_nics(self, method, url, body, headers):
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_nics.xml')
        return (httplib.OK, response, {}, '')

    def _api_admin_datacenters(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('dcs.xml'), {}, '')

    def _api_admin_enterprises_1(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ent_1.xml'), {}, '')

    def _api_admin_enterprises_1_datacenterrepositories(self, method, url, body, headers):
        if headers['Authorization'] == 'Basic Z286dHJ1bmtz':
            expected_response = self.fixtures.load('not_found_error.xml')
            return (httplib.NOT_FOUND, expected_response, {}, '')
        elif headers['Authorization'] != 'Basic c29uOmdvaGFu':
            return (httplib.OK, self.fixtures.load('ent_1_dcreps.xml'), {}, '')
        else:
            expected_response = self.fixtures.load('privilege_errors.html')
            return (httplib.FORBIDDEN, expected_response, {}, '')

    def _api_admin_enterprises_1_datacenterrepositories_2(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ent_1_dcrep_2.xml'), {}, '')

    def _api_admin_enterprises_1_datacenterrepositories_2_virtualmachinetemplates(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ent_1_dcrep_2_templates.xml'), {}, '')

    def _api_admin_enterprises_1_datacenterrepositories_2_virtualmachinetemplates_11(self, method, url, body, headers):
        return (httplib.OK, self.fixtures.load('ent_1_dcrep_2_template_11.xml'), {}, '')