from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
def test_setproctitle_brief(self):
    with mock.patch('setproctitle.setproctitle') as spt:
        _ProcWorker(set_proctitle='brief').start(name='foo', desc='bar')
        self.assertEqual(spt.call_args[0][0], 'foo: bar')