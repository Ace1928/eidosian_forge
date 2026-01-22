from unittest import mock
import testtools
from osprofiler import initializer
class InitializerTestCase(testtools.TestCase):

    @mock.patch('osprofiler.notifier.set')
    @mock.patch('osprofiler.notifier.create')
    @mock.patch('osprofiler.web.enable')
    def test_initializer(self, web_enable_mock, notifier_create_mock, notifier_set_mock):
        conf = mock.Mock()
        conf.profiler.connection_string = 'driver://'
        conf.profiler.hmac_keys = 'hmac_keys'
        context = {}
        project = 'my-project'
        service = 'my-service'
        host = 'my-host'
        notifier_mock = mock.Mock()
        notifier_create_mock.return_value = notifier_mock
        initializer.init_from_conf(conf, context, project, service, host)
        notifier_create_mock.assert_called_once_with('driver://', context=context, project=project, service=service, host=host, conf=conf)
        notifier_set_mock.assert_called_once_with(notifier_mock)
        web_enable_mock.assert_called_once_with('hmac_keys')