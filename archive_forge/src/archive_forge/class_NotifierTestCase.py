from unittest import mock
from osprofiler import notifier
from osprofiler.tests import test
class NotifierTestCase(test.TestCase):

    def tearDown(self):
        notifier.set(notifier._noop_notifier)
        notifier.clear_notifier_cache()
        super(NotifierTestCase, self).tearDown()

    def test_set(self):

        def test(info):
            pass
        notifier.set(test)
        self.assertEqual(notifier.get(), test)

    def test_get_default_notifier(self):
        self.assertEqual(notifier.get(), notifier._noop_notifier)

    def test_notify(self):
        m = mock.MagicMock()
        notifier.set(m)
        notifier.notify(10)
        m.assert_called_once_with(10)

    @mock.patch('osprofiler.notifier.base.get_driver')
    def test_create(self, mock_factory):
        result = notifier.create('test', 10, b=20)
        mock_factory.assert_called_once_with('test', 10, b=20)
        self.assertEqual(mock_factory.return_value.notify, result)

    @mock.patch('osprofiler.notifier.base.get_driver')
    def test_create_driver_init_failure(self, mock_get_driver):
        mock_get_driver.side_effect = Exception()
        result = notifier.create('test', 10, b=20)
        mock_get_driver.assert_called_once_with('test', 10, b=20)
        self.assertEqual(notifier._noop_notifier, result)