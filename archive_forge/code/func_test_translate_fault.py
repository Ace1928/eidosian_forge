from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_translate_fault(self):

    def fake_task(fault_class_name, error_msg=None):
        task_info = mock.Mock()
        task_info.localizedMessage = error_msg
        if fault_class_name:
            error_fault = mock.Mock()
            error_fault.__class__.__name__ = fault_class_name
            task_info.fault = error_fault
        return task_info
    error_msg = 'OUCH'
    task = fake_task(exceptions.FILE_LOCKED, error_msg)
    actual = exceptions.translate_fault(task)
    expected = exceptions.FileLockedException(error_msg)
    self.assertEqual(expected.__class__, actual.__class__)
    self.assertEqual(expected.message, actual.message)
    error_msg = 'Oopsie'
    task = fake_task(None, error_msg)
    actual = exceptions.translate_fault(task)
    expected = exceptions.VimFaultException(['Mock'], message=error_msg)
    self.assertEqual(expected.__class__, actual.__class__)
    self.assertEqual(expected.message, actual.message)
    self.assertEqual(expected.fault_list, actual.fault_list)