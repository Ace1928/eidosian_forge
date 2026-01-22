from unittest import mock
from osprofiler.drivers import base
from osprofiler.tests import test
class MessagingTestCase(test.TestCase):

    @mock.patch('oslo_utils.importutils.try_import')
    def test_init_no_oslo_messaging(self, try_import_mock):
        try_import_mock.return_value = None
        self.assertRaises(ValueError, base.get_driver, 'messaging://', project='project', service='service', host='host', context={})

    @mock.patch('oslo_utils.importutils.try_import')
    def test_init_and_notify(self, try_import_mock):
        context = 'context'
        transport = 'transport'
        project = 'project'
        service = 'service'
        host = 'host'
        oslo_messaging_mock = mock.Mock()
        try_import_mock.return_value = oslo_messaging_mock
        notifier_mock = mock.Mock()
        oslo_messaging_mock.Notifier.return_value = notifier_mock
        oslo_messaging_mock.get_notification_transport.return_value = transport
        notify_func = base.get_driver('messaging://', project=project, service=service, context=context, host=host).notify
        oslo_messaging_mock.Notifier.assert_called_once_with(transport, publisher_id=host, driver='messaging', topics=['profiler'], retry=0)
        info = {'a': 10, 'project': project, 'service': service, 'host': host}
        notify_func(info)
        notifier_mock.info.assert_called_once_with(context, 'profiler.service', info)
        notifier_mock.reset_mock()
        notify_func(info, context='my_context')
        notifier_mock.info.assert_called_once_with('my_context', 'profiler.service', info)