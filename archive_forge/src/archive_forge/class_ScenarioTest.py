from oslo_config import cfg
from heat.common import context
from heat.engine import resource
from heat.tests import common
from heat.tests.convergence.framework import fake_resource
from heat.tests.convergence.framework import processes
from heat.tests.convergence.framework import scenario
from heat.tests.convergence.framework import testutils
class ScenarioTest(common.HeatTestCase):
    scenarios = [(name, {'name': name, 'path': path}) for name, path in scenario.list_all()]

    def setUp(self):
        super(ScenarioTest, self).setUp()
        self.patchobject(context, 'StoredContext')
        resource._register_class('OS::Heat::TestResource', fake_resource.TestResource)
        self.procs = processes.Processes()
        po = self.patch('heat.rpc.worker_client.WorkerClient.check_resource')
        po.side_effect = self.procs.worker.check_resource
        cfg.CONF.set_default('convergence_engine', True)

    def test_scenario(self):
        self.procs.clear()
        runner = scenario.Scenario(self.name, self.path)
        runner(self.procs.event_loop, **testutils.scenario_globals(self.procs, self))