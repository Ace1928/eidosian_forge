import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.skipif(LOG_MODE == 'PYTHON', reason='Requires KIVY_LOG_MODE!=PYTHON to run.')
def test_trace_level_has_level_name():
    from kivy.logger import Logger, LoggerHistory
    Logger.setLevel(9)
    Logger.trace('test: This is trace message 1')
    assert LoggerHistory.history[0].levelname == 'TRACE'