import logging
import os
import sys
import param
from holoviews.element.comparison import ComparisonTestCase
class LoggingComparisonTestCase(ComparisonTestCase):
    """
    ComparisonTestCase with support for capturing param logging output.

    Subclasses must call super setUp to make the
    tests independent. Testing can then be done via the
    self.log_handler.tail and self.log_handler.assertEndsWith methods.
    """

    def setUp(self):
        super().setUp()
        log = param.parameterized.get_logger()
        self.handlers = log.handlers
        log.handlers = []
        self.log_handler = MockLoggingHandler(level='DEBUG')
        log.addHandler(self.log_handler)

    def tearDown(self):
        super().tearDown()
        log = param.parameterized.get_logger()
        log.handlers = self.handlers
        messages = self.log_handler.messages
        self.log_handler.reset()
        for level, msgs in messages.items():
            for msg in msgs:
                log.log(LEVELS[level], msg)