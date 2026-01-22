from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
class TestFixture(base.BaseTestCase):

    def setUp(self):
        super(TestFixture, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='oslo_limit', endpoint_id='ENDPOINT_ID')
        opts.register_opts(CONF)
        reglimits = {'widgets': 100, 'sprockets': 50}
        projlimits = {'project2': {'widgets': 10}}
        self.useFixture(fixture.LimitFixture(reglimits, projlimits))
        self.usage = {'project1': {'sprockets': 10, 'widgets': 10}, 'project2': {'sprockets': 3, 'widgets': 3}}

        def proj_usage(project_id, resource_names):
            return self.usage[project_id]
        self.enforcer = limit.Enforcer(proj_usage)

    def test_project_under_registered_limit_only(self):
        self.enforcer.enforce('project1', {'sprockets': 1, 'widgets': 1})

    def test_project_over_registered_limit_only(self):
        self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project1', {'sprockets': 1, 'widgets': 102})

    def test_project_over_registered_limit(self):
        self.enforcer.enforce('project2', {'sprockets': 1})
        self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project2', {'sprockets': 50})

    def test_project_over_project_limits(self):
        self.enforcer.enforce('project2', {'widgets': 7})
        self.assertRaises(exception.ProjectOverLimit, self.enforcer.enforce, 'project2', {'widgets': 10})

    def test_calculate_usage(self):
        u = self.enforcer.calculate_usage('project2', ['widgets'])['widgets']
        self.assertEqual(3, u.usage)
        self.assertEqual(10, u.limit)
        u = self.enforcer.calculate_usage('project1', ['widgets', 'sprockets'])
        self.assertEqual(10, u['sprockets'].usage)
        self.assertEqual(10, u['widgets'].usage)
        self.assertEqual(50, u['sprockets'].limit)
        self.assertEqual(100, u['widgets'].limit)