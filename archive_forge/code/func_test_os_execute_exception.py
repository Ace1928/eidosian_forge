from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick.caches import opencas
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
def test_os_execute_exception(self, mock_execute):
    raise_err = [putils.ProcessExecutionError(exit_code=1), mock.DEFAULT]
    engine = opencas.OpenCASEngine(root_helper=None, opencas_cache_id=1)
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine.os_execute, 'cmd', 'param')
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine.is_engine_ready)
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine._get_mapped_casdev, 'path')
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine._get_mapped_coredev, 'path')
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine._map_casdisk, 'path')
    mock_execute.side_effect = raise_err
    self.assertRaises(putils.ProcessExecutionError, engine._unmap_casdisk, 'path')