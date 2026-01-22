from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def _create_api_session(self, _create_session, retry_count=10, task_poll_interval=1):
    return api.VMwareAPISession(VMwareAPISessionTest.SERVER_IP, VMwareAPISessionTest.USERNAME, VMwareAPISessionTest.PASSWORD, retry_count, task_poll_interval, 'https', _create_session, port=VMwareAPISessionTest.PORT, cacert=self.cert_mock, insecure=False, pool_size=VMwareAPISessionTest.POOL_SIZE)