import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def wait_for_task_side_effect(task):
    task_info = mock.Mock()
    task_info.result = pg_moref
    return task_info