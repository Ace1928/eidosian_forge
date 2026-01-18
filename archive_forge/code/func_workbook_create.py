import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def workbook_create(self, wb_def, namespace='', admin=True, scope='private'):
    params = '{0}'.format(wb_def)
    namespace_params = ''
    if namespace:
        namespace_params += ' --namespace {}'.format(namespace)
    if scope == 'public':
        params += ' --public'
    params += namespace_params
    wb = self.mistral_cli(admin, 'workbook-create', params=params)
    wb_name = self.get_field_value(wb, 'Name')
    wb_delete_params = wb_name + namespace_params
    self.addCleanup(self.mistral_cli, admin, 'workbook-delete', params=wb_delete_params)
    self.addCleanup(self.mistral_cli, admin, 'workflow-delete', params='wb.wf1')
    return wb