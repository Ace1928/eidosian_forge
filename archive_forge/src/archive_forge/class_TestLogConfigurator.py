import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
class TestLogConfigurator(utils.TestCase):

    def setUp(self):
        super(TestLogConfigurator, self).setUp()
        self.options = mock.Mock()
        self.options.verbose_level = 1
        self.options.log_file = None
        self.options.debug = False
        self.root_logger = mock.Mock()
        self.root_logger.setLevel = mock.Mock()
        self.root_logger.addHandler = mock.Mock()
        self.requests_log = mock.Mock()
        self.requests_log.setLevel = mock.Mock()
        self.cliff_log = mock.Mock()
        self.cliff_log.setLevel = mock.Mock()
        self.stevedore_log = mock.Mock()
        self.stevedore_log.setLevel = mock.Mock()
        self.iso8601_log = mock.Mock()
        self.iso8601_log.setLevel = mock.Mock()
        self.loggers = [self.root_logger, self.requests_log, self.cliff_log, self.stevedore_log, self.iso8601_log]

    @mock.patch('logging.StreamHandler')
    @mock.patch('logging.getLogger')
    @mock.patch('osc_lib.logs.set_warning_filter')
    def test_init(self, warning_filter, getLogger, handle):
        getLogger.side_effect = self.loggers
        console_logger = mock.Mock()
        console_logger.setFormatter = mock.Mock()
        console_logger.setLevel = mock.Mock()
        handle.return_value = console_logger
        configurator = logs.LogConfigurator(self.options)
        getLogger.assert_called_with('iso8601')
        warning_filter.assert_called_with(logging.WARNING)
        self.root_logger.setLevel.assert_called_with(logging.DEBUG)
        self.root_logger.addHandler.assert_called_with(console_logger)
        self.requests_log.setLevel.assert_called_with(logging.ERROR)
        self.cliff_log.setLevel.assert_called_with(logging.ERROR)
        self.stevedore_log.setLevel.assert_called_with(logging.ERROR)
        self.iso8601_log.setLevel.assert_called_with(logging.ERROR)
        self.assertFalse(configurator.dump_trace)

    @mock.patch('logging.getLogger')
    @mock.patch('osc_lib.logs.set_warning_filter')
    def test_init_no_debug(self, warning_filter, getLogger):
        getLogger.side_effect = self.loggers
        self.options.debug = True
        configurator = logs.LogConfigurator(self.options)
        warning_filter.assert_called_with(logging.DEBUG)
        self.requests_log.setLevel.assert_called_with(logging.DEBUG)
        self.assertTrue(configurator.dump_trace)

    @mock.patch('logging.FileHandler')
    @mock.patch('logging.getLogger')
    @mock.patch('osc_lib.logs.set_warning_filter')
    @mock.patch('osc_lib.logs._FileFormatter')
    def test_init_log_file(self, formatter, warning_filter, getLogger, handle):
        getLogger.side_effect = self.loggers
        self.options.log_file = '/tmp/log_file'
        file_logger = mock.Mock()
        file_logger.setFormatter = mock.Mock()
        file_logger.setLevel = mock.Mock()
        handle.return_value = file_logger
        mock_formatter = mock.Mock()
        formatter.return_value = mock_formatter
        logs.LogConfigurator(self.options)
        handle.assert_called_with(filename=self.options.log_file)
        self.root_logger.addHandler.assert_called_with(file_logger)
        file_logger.setFormatter.assert_called_with(mock_formatter)
        file_logger.setLevel.assert_called_with(logging.WARNING)

    @mock.patch('logging.FileHandler')
    @mock.patch('logging.getLogger')
    @mock.patch('osc_lib.logs.set_warning_filter')
    @mock.patch('osc_lib.logs._FileFormatter')
    def test_configure(self, formatter, warning_filter, getLogger, handle):
        getLogger.side_effect = self.loggers
        configurator = logs.LogConfigurator(self.options)
        cloud_config = mock.Mock()
        config_log = '/tmp/config_log'
        cloud_config.config = {'log_file': config_log, 'verbose_level': 1, 'log_level': 'info'}
        file_logger = mock.Mock()
        file_logger.setFormatter = mock.Mock()
        file_logger.setLevel = mock.Mock()
        handle.return_value = file_logger
        mock_formatter = mock.Mock()
        formatter.return_value = mock_formatter
        configurator.configure(cloud_config)
        warning_filter.assert_called_with(logging.INFO)
        handle.assert_called_with(filename=config_log)
        self.root_logger.addHandler.assert_called_with(file_logger)
        file_logger.setFormatter.assert_called_with(mock_formatter)
        file_logger.setLevel.assert_called_with(logging.INFO)
        self.assertFalse(configurator.dump_trace)