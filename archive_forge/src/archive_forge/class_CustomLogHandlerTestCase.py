import logging
from oslo_log import log
from oslo_log.tests.unit.test_log import LogTestBase
class CustomLogHandlerTestCase(LogTestBase):

    def setUp(self):
        super(CustomLogHandlerTestCase, self).setUp()
        self.config(logging_context_format_string='HAS CONTEXT [%(request_id)s]: %(message)s', logging_default_format_string='NOCTXT: %(message)s', logging_debug_format_suffix='--DBG')
        self.log = log.getLogger('')
        self._add_handler_with_cleanup(self.log, handler=CustomLogHandler)
        self._set_log_level_with_cleanup(self.log, logging.DEBUG)

    def test_log(self):
        message = 'foo'
        self.log.info(message)
        self.assertEqual('NOCTXT: %s\n' % message, self.stream.getvalue())