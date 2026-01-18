import pytest
import logging
from charset_normalizer.utils import set_logging_handler
from charset_normalizer.api import from_bytes, explain_handler
from charset_normalizer.constant import TRACE
def test_set_stream_handler(self, caplog):
    set_logging_handler('charset_normalizer', level=logging.DEBUG)
    self.logger.debug('log content should log with default format')
    for record in caplog.records:
        assert record.levelname in ['Level 5', 'DEBUG']
    assert 'log content should log with default format' in caplog.text