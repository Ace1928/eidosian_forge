import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class TargetCallableTestCase(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(attrs=dict(), kwargs=dict(), vals=dict())), ('exchange_attr', dict(attrs=dict(exchange='testexchange'), kwargs=dict(), vals=dict(exchange='testexchange'))), ('exchange_arg', dict(attrs=dict(), kwargs=dict(exchange='testexchange'), vals=dict(exchange='testexchange'))), ('topic_attr', dict(attrs=dict(topic='testtopic'), kwargs=dict(), vals=dict(topic='testtopic'))), ('topic_arg', dict(attrs=dict(), kwargs=dict(topic='testtopic'), vals=dict(topic='testtopic'))), ('namespace_attr', dict(attrs=dict(namespace='testnamespace'), kwargs=dict(), vals=dict(namespace='testnamespace'))), ('namespace_arg', dict(attrs=dict(), kwargs=dict(namespace='testnamespace'), vals=dict(namespace='testnamespace'))), ('version_attr', dict(attrs=dict(version='3.4'), kwargs=dict(), vals=dict(version='3.4'))), ('version_arg', dict(attrs=dict(), kwargs=dict(version='3.4'), vals=dict(version='3.4'))), ('server_attr', dict(attrs=dict(server='testserver'), kwargs=dict(), vals=dict(server='testserver'))), ('server_arg', dict(attrs=dict(), kwargs=dict(server='testserver'), vals=dict(server='testserver'))), ('fanout_attr', dict(attrs=dict(fanout=True), kwargs=dict(), vals=dict(fanout=True))), ('fanout_arg', dict(attrs=dict(), kwargs=dict(fanout=True), vals=dict(fanout=True)))]

    def test_callable(self):
        target = oslo_messaging.Target(**self.attrs)
        target = target(**self.kwargs)
        for k in self.vals:
            self.assertEqual(self.vals[k], getattr(target, k))
        for k in ['exchange', 'topic', 'namespace', 'version', 'server', 'fanout']:
            if k in self.vals:
                continue
            self.assertIsNone(getattr(target, k))