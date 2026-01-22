import logging
import os
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import initializer
from osprofiler import opts
from osprofiler import profiler
from osprofiler.tests import test
class DriverTestCase(test.FunctionalTestCase):
    SERVICE = 'service'
    PROJECT = 'project'

    def setUp(self):
        super(DriverTestCase, self).setUp()
        CONF(['--config-file', os.path.dirname(__file__) + '/config.cfg'])
        opts.set_defaults(CONF, enabled=True, trace_sqlalchemy=False, hmac_keys='SECRET_KEY')

    def _assert_dict(self, info, **kwargs):
        for key in kwargs:
            self.assertEqual(kwargs[key], info[key])

    def _assert_child_dict(self, child, base_id, parent_id, name, fn_name):
        self.assertEqual(parent_id, child['parent_id'])
        exp_info = {'name': 'rpc', 'service': self.SERVICE, 'project': self.PROJECT}
        self._assert_dict(child['info'], **exp_info)
        raw_start = child['info']['meta.raw_payload.%s-start' % name]
        self.assertEqual(fn_name, raw_start['info']['function']['name'])
        exp_raw = {'name': '%s-start' % name, 'service': self.SERVICE, 'trace_id': child['trace_id'], 'project': self.PROJECT, 'base_id': base_id}
        self._assert_dict(raw_start, **exp_raw)
        raw_stop = child['info']['meta.raw_payload.%s-stop' % name]
        exp_raw['name'] = '%s-stop' % name
        self._assert_dict(raw_stop, **exp_raw)

    def test_get_report(self):
        initializer.init_from_conf(CONF, {}, self.PROJECT, self.SERVICE, 'host')
        profiler.init('SECRET_KEY')
        base_id = profiler.get().get_base_id()
        foo = Foo()
        foo.bar(1)
        engine = base.get_driver(CONF.profiler.connection_string, project=self.PROJECT, service=self.SERVICE, host='host', conf=CONF)
        report = engine.get_report(base_id)
        LOG.debug('OSProfiler report: %s', report)
        self.assertEqual('total', report['info']['name'])
        self.assertEqual(2, report['stats']['rpc']['count'])
        self.assertEqual(1, len(report['children']))
        cbar = report['children'][0]
        self._assert_child_dict(cbar, base_id, base_id, 'rpc', 'osprofiler.tests.functional.test_driver.Foo.bar')
        self.assertEqual(1, len(cbar['children']))
        cbaz = cbar['children'][0]
        self._assert_child_dict(cbaz, base_id, cbar['trace_id'], 'rpc', 'osprofiler.tests.functional.test_driver.Foo.baz')