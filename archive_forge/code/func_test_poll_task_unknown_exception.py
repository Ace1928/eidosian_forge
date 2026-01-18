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
def test_poll_task_unknown_exception(self):
    _unknown_exceptions = {'NotAFile': exceptions.VimFaultException, 'RuntimeFault': exceptions.VimFaultException}
    for k, v in _unknown_exceptions.items():
        self._poll_task_well_known_exceptions(k, v)