from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
def test_periodic_tasks_with_idle(self):

    class Manager(periodic_task.PeriodicTasks):

        @periodic_task.periodic_task(spacing=200)
        def bar(self):
            return 'bar'
    m = Manager(self.conf)
    self.assertThat(m._periodic_tasks, matchers.HasLength(1))
    self.assertEqual(200, m._periodic_spacing['bar'])
    idle = m.run_periodic_tasks(None)
    self.assertAlmostEqual(60, idle, 1)