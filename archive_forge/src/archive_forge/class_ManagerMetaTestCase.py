from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
class ManagerMetaTestCase(base.ServiceBaseTestCase):
    """Tests for the meta class which manages creation of periodic tasks."""

    def test_meta(self):

        class Manager(periodic_task.PeriodicTasks):

            @periodic_task.periodic_task
            def foo(self):
                return 'foo'

            @periodic_task.periodic_task(spacing=4)
            def bar(self):
                return 'bar'

            @periodic_task.periodic_task(enabled=False)
            def baz(self):
                return 'baz'
        m = Manager(self.conf)
        self.assertThat(m._periodic_tasks, matchers.HasLength(2))
        self.assertEqual(periodic_task.DEFAULT_INTERVAL, m._periodic_spacing['foo'])
        self.assertEqual(4, m._periodic_spacing['bar'])
        self.assertThat(m._periodic_spacing, matchers.Not(matchers.Contains('baz')))

        @periodic_task.periodic_task
        def external():
            return 42
        m.add_periodic_task(external)
        self.assertThat(m._periodic_tasks, matchers.HasLength(3))
        self.assertEqual(periodic_task.DEFAULT_INTERVAL, m._periodic_spacing['external'])