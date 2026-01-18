import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def workflow_create(self, wf_def, namespace='', admin=True, scope='private'):
    params = '{0}'.format(wf_def)
    if scope == 'public':
        params += ' --public'
    if namespace:
        params += ' --namespace ' + namespace
    wf = self.mistral_cli(admin, 'workflow-create', params=params)
    for workflow in wf:
        self.addCleanup(self.mistral_cli, admin, 'workflow-delete', params=workflow['ID'])
    return wf