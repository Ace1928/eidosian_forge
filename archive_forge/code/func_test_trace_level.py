import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.skipif(LOG_MODE != 'KIVY', reason='Requires KIVY_LOG_MODE==KIVY to run.')
def test_trace_level():
    """Kivy logger defines a custom level of Trace."""
    from kivy.logger import Logger, LOG_LEVELS, LoggerHistory
    import logging
    Logger.setLevel(9)
    Logger.trace('test: This is trace message 1')
    logging.log(logging.TRACE, 'test: This is trace message 2')
    Logger.log(LOG_LEVELS['trace'], 'test: This is trace message 3')
    last_log_records = LoggerHistory.history[:3]
    assert all((log_record.levelno == 9 for log_record in last_log_records)), [log_record.levelno for log_record in last_log_records]