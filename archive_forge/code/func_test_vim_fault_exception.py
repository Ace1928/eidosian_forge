from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_vim_fault_exception(self):
    vfe = exceptions.VimFaultException([ValueError('example')], _('cause'))
    string = str(vfe)
    self.assertIn(string, ["cause\nFaults: [ValueError('example',)]", "cause\nFaults: [ValueError('example')]"])