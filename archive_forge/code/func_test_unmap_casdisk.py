from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick.caches import opencas
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
def test_unmap_casdisk(self, moc_exec):
    engine = opencas.OpenCASEngine(root_helper=None, opencas_cache_id=1)
    moc_exec.return_value = ('', '')
    engine._unmap_casdisk('1')
    moc_exec.assert_has_calls([mock.call('casadm', '-R', '-f', '-i', 1, '-j', '1', run_as_root=True, root_helper=None)])