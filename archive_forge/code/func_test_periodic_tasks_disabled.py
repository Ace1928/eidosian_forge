from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
def test_periodic_tasks_disabled(self):

    class Manager(periodic_task.PeriodicTasks):

        @periodic_task.periodic_task(spacing=-1)
        def bar(self):
            return 'bar'
    m = Manager(self.conf)
    idle = m.run_periodic_tasks(None)
    self.assertAlmostEqual(60, idle, 1)