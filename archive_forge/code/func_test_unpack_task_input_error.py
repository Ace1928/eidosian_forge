from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_unpack_task_input_error(self):
    task_input1 = {'import_from_format': 'bar', 'image_properties': 'baz'}
    task_input2 = {'import_from': 'foo', 'image_properties': 'baz'}
    task_input3 = {'import_from': 'foo', 'import_from_format': 'bar'}
    task1 = mock.Mock(task_input=task_input1)
    task2 = mock.Mock(task_input=task_input2)
    task3 = mock.Mock(task_input=task_input3)
    self.assertRaises(exception.Invalid, script_utils.unpack_task_input, task1)
    self.assertRaises(exception.Invalid, script_utils.unpack_task_input, task2)
    self.assertRaises(exception.Invalid, script_utils.unpack_task_input, task3)