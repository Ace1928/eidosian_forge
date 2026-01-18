from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
def test_proctitle_custom_name(self):
    with mock.patch('setproctitle.setproctitle') as spt:
        _ProcWorker().start(name='tardis')
        self.assertRegex(spt.call_args[0][0], '^tardis: _ProcWorker \\(.*python.*\\)$')