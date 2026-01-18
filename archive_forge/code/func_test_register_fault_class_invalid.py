from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_register_fault_class_invalid(self):
    self.assertRaises(TypeError, exceptions.register_fault_class, 'ValueError', ValueError)